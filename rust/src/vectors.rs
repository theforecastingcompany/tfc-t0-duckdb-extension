//! Typed reads from DuckDB's nested C vectors, isolated from model execution.
//!
//! The pinned duckdb crate's Arrow input conversion does not support LIST or
//! STRUCT. Read the explicit batch directly, avoiding an intermediate Arrow copy.
use duckdb::{
    core::{DataChunkHandle, FlatVector, ListVector, LogicalTypeId},
    ffi::duckdb_list_entry,
    vtab::arrow::WritableVector,
};
use duckdb_ext_core::{ForecastError, ForecastInput};
use std::ops::Range;

pub(super) fn read_batch(
    input: &DataChunkHandle,
    row: usize,
) -> Result<Vec<ForecastInput>, ForecastError> {
    if input.num_columns() != 2 || row >= input.len() {
        return Err("expected a request list and horizon".into());
    }
    let batches = input.flat_vector(0);
    let horizons = input.flat_vector(1);
    if input.len() > batches.capacity() || input.len() > horizons.capacity() {
        return Err("DuckDB input exceeds vector capacity".into());
    }
    if horizons.logical_type().id() != LogicalTypeId::Bigint {
        return Err("horizon must be BIGINT".into());
    }
    if horizons.row_is_null(row as u64) {
        return Err("horizon must not be NULL".into());
    }
    // SAFETY: checked BIGINT storage, capacity and validity. This initialized
    // input belongs to DuckDB and is read only during the callback.
    let horizon = unsafe { horizons.as_mut_ptr::<i64>().add(row).read() };
    let horizon = usize::try_from(horizon).map_err(|_| "horizon must be 1..1024")?;
    let kind = batches.logical_type();
    if kind.id() != LogicalTypeId::List || kind.child(0).id() != LogicalTypeId::Struct {
        return Err("batch must be a list of request structs".into());
    }
    let fields = kind.child(0);
    if fields.num_children() != 3 {
        return Err("requests must contain history, covariate_history and covariate_future".into());
    }
    for (column, name) in ["history", "covariate_history", "covariate_future"]
        .into_iter()
        .enumerate()
    {
        let field = fields.child(column);
        if fields.child_name(column) != name
            || field.id() != LogicalTypeId::List
            || field.child(0).id() != LogicalTypeId::Float
        {
            return Err(
                "request fields must be history, covariate_history and covariate_future FLOAT[]"
                    .into(),
            );
        }
    }
    let list = input.list_vector(0);
    let range = list_range(&batches, &list, row)?;
    if range.is_empty() {
        return Err("batch must contain at least one series".into());
    }
    let count = list.len();
    let structs = list.struct_child(count);
    let validity = list.child(count);
    let columns = (0..3)
        .map(|column| {
            (
                structs.child(column, count),
                structs.list_vector_child(column),
            )
        })
        .collect::<Vec<_>>();
    let mut requests = Vec::with_capacity(range.len());
    for index in range {
        if validity.row_is_null(index as u64) {
            return Err("batch requests must not be NULL".into());
        }
        requests.push(ForecastInput::new(
            values(&columns[0].0, &columns[0].1, index)?,
            horizon,
            values(&columns[1].0, &columns[1].1, index)?,
            values(&columns[2].0, &columns[2].1, index)?,
        )?);
    }
    Ok(requests)
}

fn list_range(
    entries: &FlatVector<'_>,
    list: &ListVector<'_>,
    row: usize,
) -> Result<Range<usize>, ForecastError> {
    if row >= entries.capacity() || entries.logical_type().id() != LogicalTypeId::List {
        return Err("invalid DuckDB list row".into());
    }
    if entries.row_is_null(row as u64) {
        return Err("batch, history and covariate lists must not be NULL".into());
    }
    // SAFETY: checked LIST storage, actual parent capacity and validity. Read
    // only this initialized entry, without borrowing a mutable slice. Nested
    // lists can exceed DuckDB's standard chunk size, so use their parent length
    // rather than ListVector::get_entry's default-sized slice.
    let entry = unsafe { entries.as_mut_ptr::<duckdb_list_entry>().add(row).read() };
    let offset = usize::try_from(entry.offset)?;
    let length = usize::try_from(entry.length)?;
    let end = offset
        .checked_add(length)
        .filter(|&end| end <= list.len())
        .ok_or("invalid DuckDB list bounds")?;
    Ok(offset..end)
}

fn values(
    entries: &FlatVector<'_>,
    list: &ListVector<'_>,
    row: usize,
) -> Result<Vec<f32>, ForecastError> {
    let range = list_range(entries, list, row)?;
    let child = list.child(list.len());
    let mut values = Vec::with_capacity(range.len());
    for index in range {
        if child.row_is_null(index as u64) {
            return Err("history and covariates contain NULL values".into());
        }
        // SAFETY: read_batch checked LIST<FLOAT> storage; list_range checked
        // child bounds, and this element is valid. No pointer escapes or input
        // mutation occurs while reading this initialized f32.
        values.push(unsafe { child.as_mut_ptr::<f32>().add(index).read() });
    }
    Ok(values)
}

/// Write nested results with capacity based on series count, not chunk size.
/// The upstream nested Arrow writer uses default-sized inner list entries.
pub(super) fn write_batches(
    predictions: &[Vec<Vec<f32>>],
    output: &mut dyn WritableVector,
) -> Result<(), ForecastError> {
    if predictions.len() > output.flat_vector().capacity() {
        return Err("output exceeds DuckDB vector capacity".into());
    }
    let series_count = predictions.iter().try_fold(0_usize, |n, batch| {
        n.checked_add(batch.len()).ok_or("too many output series")
    })?;
    let value_count = predictions
        .iter()
        .flatten()
        .try_fold(0_usize, |n, series| {
            n.checked_add(series.len()).ok_or("too many output values")
        })?;
    let mut batches = output.list_vector();
    // Reserve both child allocations before taking any data pointers.
    let entries = batches.child(series_count);
    batches.set_len(series_count);
    let series = batches.list_child();
    let values = series.child(value_count);
    series.set_len(value_count);
    let mut series_offset = 0;
    let mut value_offset = 0;
    for (row, batch) in predictions.iter().enumerate() {
        batches.set_entry(row, series_offset, batch.len());
        for prediction in batch {
            // SAFETY: the registered output is FLOAT[][]; the allocations above
            // cover every entry/value written here. No reallocations or aliased
            // references occur while writing these disjoint initialized ranges.
            unsafe {
                entries
                    .as_mut_ptr::<duckdb_list_entry>()
                    .add(series_offset)
                    .write(duckdb_list_entry {
                        offset: value_offset as u64,
                        length: prediction.len() as u64,
                    });
                if !prediction.is_empty() {
                    std::ptr::copy_nonoverlapping(
                        prediction.as_ptr(),
                        values.as_mut_ptr::<f32>().add(value_offset),
                        prediction.len(),
                    );
                }
            }
            series_offset += 1;
            value_offset += prediction.len();
        }
    }
    Ok(())
}

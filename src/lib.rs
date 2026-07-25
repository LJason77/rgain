/*
 * Copyright (c) 2026, LJason. All Rights Reserved.
 */

mod error;
mod file_ops;
mod pipeline;

pub use error::{AppError, AppResult};
pub use pipeline::{run_pipeline, TrackResult};

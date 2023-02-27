use polars::prelude::*;

pub fn sanitize_df(df: &mut Arc<DataFrame>, blacklist: &Vec<String>) {
    let df = Arc::make_mut(df);
    for name in blacklist {
        let idx = match df.get_column_names().iter().position(|x| x == name) {
            Some(idx) => idx,
            None => continue,
        };
        let series = df.get_columns_mut().get_mut(idx).unwrap();
        *series = Series::full_null(name, series.len(), series.dtype());
    }
}

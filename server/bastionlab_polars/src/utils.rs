use polars::prelude::*;

pub fn sanitize_df(df: &mut DataFrame, blacklist: &Vec<String>) {
    for name in blacklist {
        let idx = match df.get_column_names().iter().position(|x| x == name) {
            Some(idx) => idx,
            None => continue,
        };
        let series = df.get_columns_mut().get_mut(idx).unwrap();
        *series = Series::full_null(name, series.len(), series.dtype());
    }
}

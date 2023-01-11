use linfa_linear::LinearRegression;

pub fn linear_regression(fit_intercept: bool) -> LinearRegression {
    let model = LinearRegression::new();
    let model = model.with_intercept(fit_intercept);

    model
}

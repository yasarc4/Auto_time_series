# SSB Cash in Transit - Cash Demand Forecasting

### What we Forecast?
  > ATM/BTM/CRS cash demand using historical data and using the various machine & resource constraints to plan the cash replenishment trips.

### Method Used:
 * Aggregate to day level
 * Identify Changepoints
 * Fit trends for different changepoints
 * Identify Holiday Performance Pattern (Yet to be implemented)
 * Outlier Adjustment (Yet to be implemented)
 * Decompose Seasonality for multiple levels
 * Find Randomness
 * Identify Randomness Pattern (Yet to be implemented)
 * Combine the components to obtain fitted values

### Required Packages:
 * pandas
 * numpy
 * sklearn

### PreRequisites to run?
  > Add the dataset path to System Environment

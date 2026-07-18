import os
import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from sklearn.preprocessing import MinMaxScaler
from dataclasses import dataclass
from typing import List, Any

@dataclass
class ProcessedData:
    """Container for all processed and scaled data."""
    # Darts format data (for Darts models)
    darts_train_scaled: Any
    darts_val_scaled: Any
    darts_test_scaled: Any
    darts_scaler: Any  # Darts Scaler for inverse transform

    # llm format data (for Chronos/Toto models)
    llm_train_scaled: np.ndarray
    llm_val_scaled: np.ndarray
    llm_test_scaled: np.ndarray
    llm_scaler: Any  # MinMaxScaler for inverse transform
    
    # New: Information about the columns used for LLM models and the target ticker
    mid_price_columns: List[str] # Ordered list of 'mid_price' columns (e.g., ['mid_price_usdaus', 'mid_price_usdcny', 'mid_price_usdeur'])
    base_ticker_mid_price_col_name: str # e.g., 'mid_price_usdeur' if config path is usdeur-fx-train.csv

    # Test metadata (common to all models)
    test_fx_timestamps: List
    test_bid_prices: List[float]
    test_ask_prices: List[float]
    test_news_timestamps: List
    test_news_sentiments: List[float]
    test_mid_prices: List[float]  # Unscaled test mid prices for evaluation

class DataProcessor:
    def __init__(self, fx_trading_config):
        self.fx_trading_config = fx_trading_config

    # def load_fx_data(self):
    #     """Loads the time series data."""
    #     fx_data_train = pd.read_csv(self.fx_trading_config.FX_DATA_PATH_TRAIN)
    #     fx_data_val = pd.read_csv(self.fx_trading_config.FX_DATA_PATH_VAL)
    #     fx_data_test = pd.read_csv(self.fx_trading_config.FX_DATA_PATH_TEST)

    #     for df in [fx_data_train, fx_data_val, fx_data_test]:
    #         df["date"] = pd.to_datetime(df["date"], errors="coerce")
    #         df.sort_values("date", inplace=True)
    #         df.reset_index(drop=True, inplace=True)

    #     return fx_data_train, fx_data_val, fx_data_test

    def load_fx_data(self):
        # Helper function to dynamically extract the ticker name from any file path
        # example: "usdeur-fx-train.csv" -> "usdeur"
        def get_ticker_suffix(path_or_filename):
            basename = os.path.basename(path_or_filename)
            name_without_ext = os.path.splitext(basename)[0]
            return name_without_ext.split("-")[0]

        def process_and_align(split_name, config_path):
            path_to_data = os.path.dirname(config_path)
            df_1_name = f"usdgbp-fx-{split_name}.csv"
            df_2_name = f"usdeur-fx-{split_name}.csv"
            
            path_1 = os.path.join(path_to_data, df_1_name)
            path_2 = os.path.join(path_to_data, df_2_name)
            
            df_config = pd.read_csv(config_path)
            df_1 = pd.read_csv(path_1)
            df_2 = pd.read_csv(path_2)
            
            for df in [df_config, df_1, df_2]:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                df.set_index("date", inplace=True)
            
            suffix_config = f"_{get_ticker_suffix(config_path)}"
            suffix_1 = f"_{get_ticker_suffix(df_1_name)}"
            suffix_2 = f"_{get_ticker_suffix(df_2_name)}"
            
            df_config = df_config.add_suffix(suffix_config)
            df_1 = df_1.add_suffix(suffix_1)
            df_2 = df_2.add_suffix(suffix_2)
            
            df_aligned = pd.concat([df_config, df_1, df_2], axis=1)
            df_aligned = df_aligned.sort_index()
            
            df_aligned = df_aligned.ffill().dropna()
            df_aligned = df_aligned.reset_index()
            
            return df_aligned

        # Process all three splits using their respective config paths
        fx_data_train = process_and_align("train", self.fx_trading_config.FX_DATA_PATH_TRAIN)
        fx_data_val = process_and_align("val", self.fx_trading_config.FX_DATA_PATH_VAL)
        fx_data_test = process_and_align("test", self.fx_trading_config.FX_DATA_PATH_TEST)
        
        return fx_data_train, fx_data_val, fx_data_test

    def load_news_data(self):
        """Loads the news data."""
        news_data_train = pd.read_csv(self.fx_trading_config.NEWS_DATA_PATH_TRAIN)
        news_data_test = pd.read_csv(self.fx_trading_config.NEWS_DATA_PATH_TEST)

        for df in [news_data_train, news_data_test]:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df.sort_values("date", inplace=True)
            df.reset_index(drop=True, inplace=True)

        return news_data_train, news_data_test

    def _aggregate_news_by_minute(self, news_df, sentiment_col):
        """
        For timestamps that share the same year-month-day hour:minute (ignoring seconds),
        take the majority sentiment (+1, 0, -1). Returns one row per minute.
        """
        if news_df.empty:
            return news_df

        df = news_df.copy()

        # Collapse to minute precision (drops seconds)
        df["date_minute"] = df["date"].dt.floor("min")

        def pick_majority_with_tie_zero(x):
            counts = x.value_counts()
            max_count = counts.max()
            # Retrieve all sentiments tied for max count
            tied = counts[counts == max_count].index.tolist()

            if len(tied) == 1:
                return tied[0]  # Normal majority
            else:
                return 0  # No trade sentiment

        # For each minute, take the majority sentiment
        agg = (
            df.groupby("date_minute", as_index=False)[sentiment_col]
              .agg(pick_majority_with_tie_zero)
              .rename(columns={"date_minute": "date"})
        )

        # Ensure sorted by time
        agg.sort_values("date", inplace=True)
        agg.reset_index(drop=True, inplace=True)

        return agg

    def split_and_scale_data(self):
        """Split and scale data for all models."""
        input_chunk_length = self.fx_trading_config.INPUT_CHUNK_LENGTH
        fx_data_train, fx_data_val, fx_data_test = self.load_fx_data()
        news_data_train, news_data_test = self.load_news_data()

        # Aggregate news by minute with majority sentiment
        sentiment_col = self.fx_trading_config.SENTIMENT_SOURCE
        news_data_train = self._aggregate_news_by_minute(news_data_train, sentiment_col)
        news_data_test = self._aggregate_news_by_minute(news_data_test, sentiment_col)

        # 1. Dynamically identify base ticker name from config path
        # Example: "usdeur-fx-train.csv" -> "usdeur"
        config_path = self.fx_trading_config.FX_DATA_PATH_TRAIN
        base_ticker_name = os.path.splitext(os.path.basename(config_path))[0].split("-")[0]

        # Use fallback columns in case the data is ever run without suffixes
        bid_col = f"bid_price_{base_ticker_name}" if f"bid_price_{base_ticker_name}" in fx_data_test.columns else "bid_price"
        ask_col = f"ask_price_{base_ticker_name}" if f"ask_price_{base_ticker_name}" in fx_data_test.columns else "ask_price"
        
        # Extract test metadata
        fx_timestamps = fx_data_test["date"].tolist()
        bid_prices = fx_data_test[bid_col].tolist()
        ask_prices = fx_data_test[ask_col].tolist()
        news_timestamps = news_data_test["date"].tolist()
        news_sentiments = news_data_test[sentiment_col].tolist()

        # 2. Dynamically gather all mid_price columns (e.g., mid_price_usdeur, mid_price_usdaus...)
        mid_price_cols = sorted([col for col in fx_data_train.columns if col.startswith("mid_price_")])
        if not mid_price_cols: # Fallback for non-suffixed data
            mid_price_cols = ["mid_price"]

        # Determine the exact mid_price column name for the base ticker
        base_ticker_mid_price_col_name = f"mid_price_{base_ticker_name}" if f"mid_price_{base_ticker_name}" in mid_price_cols else "mid_price"
        if base_ticker_mid_price_col_name not in mid_price_cols:
            raise ValueError(f"Target mid-price column '{base_ticker_mid_price_col_name}' not found in the list of available mid-price columns: {mid_price_cols}")


        # --- Prepare Darts format data ---
        darts_train = TimeSeries.from_dataframe(fx_data_train, value_cols=mid_price_cols)
        darts_val = TimeSeries.from_dataframe(fx_data_val, value_cols=mid_price_cols)
        darts_test = TimeSeries.from_dataframe(fx_data_test, value_cols=mid_price_cols)

        # Scale Darts series (Scaler scales multivariate columns independently by default)
        darts_scaler = Scaler()
        darts_train_scaled = darts_scaler.fit_transform(darts_train)
        darts_val_scaled = darts_scaler.transform(darts_val)
        darts_test_scaled = darts_scaler.transform(darts_test)

        # --- Prepare LLM format data ---
        # Extract as a 2D numpy array of shape (n_samples, n_features) without flattening
        llm_train = fx_data_train[mid_price_cols].values.astype(np.float32)
        llm_val = fx_data_val[mid_price_cols].values.astype(np.float32)
        llm_test = fx_data_test[mid_price_cols].values.astype(np.float32)

        # Scale LLM arrays (MinMaxScaler scales each column independently when shape is 2D)
        llm_scaler = MinMaxScaler(feature_range=(0, 1))
        llm_train_scaled = llm_scaler.fit_transform(llm_train)
        llm_val_scaled = llm_scaler.transform(llm_val)
        llm_test_scaled = llm_scaler.transform(llm_test)

        # --- Prepare test metadata with input_chunk_length offset ---
        test_fx_timestamps = fx_timestamps[input_chunk_length:]
        test_bid_prices = bid_prices[input_chunk_length:]
        test_ask_prices = ask_prices[input_chunk_length:]

        # Extract unscaled test mid prices for the target trading asset
        test_mid_prices = fx_data_test[base_ticker_mid_price_col_name].values[input_chunk_length:].tolist()

        return ProcessedData(
            darts_train_scaled=darts_train_scaled,
            darts_val_scaled=darts_val_scaled,
            darts_test_scaled=darts_test_scaled,
            darts_scaler=darts_scaler,
            llm_train_scaled=llm_train_scaled,
            llm_val_scaled=llm_val_scaled,
            llm_test_scaled=llm_test_scaled,
            llm_scaler=llm_scaler,
            mid_price_columns=mid_price_cols,
            base_ticker_mid_price_col_name=base_ticker_mid_price_col_name,
            test_fx_timestamps=test_fx_timestamps,
            test_bid_prices=test_bid_prices,
            test_ask_prices=test_ask_prices,
            test_news_timestamps=news_timestamps,
            test_news_sentiments=news_sentiments,
            test_mid_prices=test_mid_prices,
        )
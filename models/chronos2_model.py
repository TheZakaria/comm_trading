# from ml_fx_trading.data_processing.data_processor import ProcessedData
from models import FinancialForecastingModel
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from chronos import BaseChronosPipeline, Chronos2Pipeline
from datetime import datetime, timedelta
class Chronos2FinancialForecastingModel(FinancialForecastingModel):
    """Financial forecasting model using Amazon's Chronos-2, predicting all variables and unscaling natively."""

    def __init__(self, model_config, data_processor):
        self.data_processor = data_processor
        self.model_config = model_config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.MODEL_NAME = "amazon/chronos-2"
        self.forecaster = self.initialize_model()

        # Placeholders to be set dynamically during prediction
        self.llm_scaler: MinMaxScaler = None
        self.target_column_index: int = -1

    def initialize_model(self):
        """Load pre-trained predictor"""
        try:
            print(f"\nLoading Chronos-2 model...")
            pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained(
                self.MODEL_NAME, 
                device_map="auto" if torch.cuda.is_available() else None
            )
            print("Chronos-2 model loaded successfully!")
            return pipeline
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error initializing Chronos model: {e}")
            raise

    def _align_test_targets(self, **test_series):
        """Process all test data series by applying the input chunk length offset."""
        return [
            series[self.model_config.INPUT_CHUNK_LENGTH:]
            for series in test_series.values()
        ]

    def train(self):
        """No training needed for zero-shot forecasting"""
        print("Model used in zero-shot mode - skipping training")
        return None

    def predict_future_values(self, input_sequences):
        """Make prediction for a batch of input sequences, returning forecasts for all features."""
        # Input shape: (batch_size, input_chunk_length, num_features)
        batch_array = np.array(input_sequences, dtype=np.float32)

        # Transpose from (batch_size, sequence_length, num_features) 
        # to (batch_size, num_features, sequence_length) to match Chronos-2 specifications
        batch_array = batch_array.transpose(0, 2, 1)

        inputs = torch.from_numpy(batch_array).float()
        try:
            with torch.no_grad():
                # quantiles: (batch_size, num_features, prediction_length, num_quantiles)
                # mean: (batch_size, num_features, prediction_length)
                quantiles, mean = self.forecaster.predict_quantiles(
                    inputs,
                    prediction_length=self.model_config.OUTPUT_CHUNK_LENGTH
                )
                preds = np.array(mean)
            return preds

        except Exception as e:
            print(f"Error during prediction: {e}")
            # Fallback: Tile the last known value for each feature across the prediction length
            last_known = input_sequences[:, -1, :, None] # Shape: (batch_size, num_features, 1)
            fallback = np.tile(last_known, (1, 1, self.model_config.OUTPUT_CHUNK_LENGTH))
            return fallback

    def generate_predictions(self, processed_data):#: ProcessedData):
        """Generate predictions using sliding window with batching for multivariate input."""
        import time
        start_time = time.time()
        print("Starting prediction generation...")

        # Extract scaling and index metadata from ProcessedData
        self.llm_scaler = processed_data.llm_scaler
        self.target_column_index = processed_data.mid_price_columns.index(processed_data.base_ticker_mid_price_col_name)

        data = processed_data.llm_test_scaled  # Shape: (num_timesteps, num_features)
        num_timesteps, num_features = data.shape
        num_predictions = num_timesteps - self.model_config.INPUT_CHUNK_LENGTH

        if num_predictions <= 0:
            raise ValueError(f"Not enough data points. Need at least {self.model_config.INPUT_CHUNK_LENGTH + 1} timesteps, got {num_timesteps}")
        
        # Preallocate 2D array to hold scaled predictions for ALL features at our target forecast horizon
        # Shape: (num_predictions, num_features)
        all_predictions_scaled = np.empty((num_predictions, num_features), dtype=np.float32)

        for batch_idx in range(0, num_predictions, self.model_config.EVAL_BATCH_SIZE):
            batch_start = batch_idx
            batch_end = min(batch_idx + self.model_config.EVAL_BATCH_SIZE, num_predictions)
            
            if batch_idx % (self.model_config.EVAL_BATCH_SIZE * 1000) == 0:
                elapsed = time.time() - start_time
                print(f"Processing batch {batch_idx} - {batch_end} / {num_predictions}. Elapsed time: {elapsed:.2f}s")
                
            # Extract multivariate sliding window sequences
            indices = np.arange(batch_start, batch_end)[:, None] + np.arange(self.model_config.INPUT_CHUNK_LENGTH)
            batch_input = data[indices] # Shape: (batch_size, INPUT_CHUNK_LENGTH, num_features)

            # Get predictions for all features: (batch_size, num_features, OUTPUT_CHUNK_LENGTH)
            predictions = self.predict_future_values(batch_input)

            # Select the final forecasting horizon step for all features
            # Shape: (batch_size, num_features)
            target_step_predictions = predictions[:, :, self.model_config.OUTPUT_CHUNK_LENGTH - 1]

            # Place predictions into our preallocated array
            all_predictions_scaled[batch_start:batch_end] = target_step_predictions

        # 1. Unscale predictions for all variables simultaneously using the scaler object natively
        all_predictions_unscaled = self.llm_scaler.inverse_transform(all_predictions_scaled)

        # 2. Extract only the target mid prices from the unscaled output array
        predicted_mid_prices = all_predictions_unscaled[:, self.target_column_index].ravel().tolist()

        return predicted_mid_prices
from models import FinancialForecastingModel
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from chronos import BaseChronosPipeline, Chronos2Pipeline
from datetime import datetime, timedelta
class Chronos2FinancialForecastingModel(FinancialForecastingModel):
    """Financial forecasting model using Amazon's Chronos-bolt, a pre-trained transformer for zero-shot forecasting"""

    def __init__(self, data_processor, model_config):
        self.data_processor = data_processor
        self.model_config = model_config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.MODEL_NAME = "s3://autogluon/chronos-2/"
        self.forecaster = self.initialize_model()

        # The scaler and target column info will be loaded when `generate_predictions` is called with ProcessedData
        self.llm_scaler: MinMaxScaler = None
        self.target_column_index: int = -1
        self.target_column_min_val: float = None
        self.target_column_scale_val: float = None


    def initialize_model(self):
        """Load pre-trained predictor"""
        try:
            print(f"\nLoading Chronos-Bolt model...")
            # Ensure proper device_map for Chronos 2
            pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained(
                self.MODEL_NAME,
                device_map="auto" if torch.cuda.is_available() else None # "auto" for multi-GPU, None for CPU
            )
            print("Chronos-Bolt model loaded successfully!")
            return pipeline
        except Exception as e:
            print(f"Error initializing Chronos model: {e}")
            raise # Re-raise to prevent silent failures

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
            """Make prediction for a batch of input sequences (multivariate input)"""

            # input_sequences will have shape (batch_size, input_chunk_length, num_features)
            batch_array = np.array(input_sequences, dtype=np.float32)

            # Chronos 2 expects input in (batch_size, num_features, sequence_length) format
            # Transpose from (batch_size, sequence_length, num_features) to (batch_size, num_features, sequence_length)
            inputs = torch.FloatTensor(batch_array.transpose(0, 2, 1)).to(self.device) 
            
            try:
                with torch.no_grad():
                    quantiles, mean = self.forecaster.predict_quantiles(
                        inputs,
                        prediction_length=self.model_config.OUTPUT_CHUNK_LENGTH
                    )
                    
                    # 'mean' will have shape (batch_size, num_features, prediction_length)
                    # We need to extract the predictions for the TARGET ticker only
                    # and then select the last step in the prediction horizon
                    
                    # Extract the target ticker's predictions across the prediction_length
                    target_ticker_predictions = np.array(mean)[:, self.target_column_index, :] # Shape: (batch_size, prediction_length)
                    
                    return target_ticker_predictions

            except Exception as e:
                print(f"Error during prediction: {e}")
                # Fallback: return the last known value for the target ticker from the input sequence
                return np.array([seq[-1, self.target_column_index] for seq in input_sequences])

    def generate_predictions(self, processed_data: ProcessedData):
        """Generate predictions using sliding window with batching for multivariate input."""
        import time
        start_time = time.time()
        print("Starting prediction generation...")

        # Initialize scaler and target column info from processed_data
        self.llm_scaler = processed_data.llm_scaler
        
        # Get the index of the target currency's mid_price column
        self.target_column_index = processed_data.mid_price_columns.index(processed_data.base_ticker_mid_price_col_name)
        
        # Get min and scale values for the target column for manual inverse transform
        self.target_column_min_val = self.llm_scaler.min_[self.target_column_index]
        self.target_column_scale_val = self.llm_scaler.scale_[self.target_column_index]

        data = processed_data.llm_test_scaled # This is (n_samples, num_features)
        num_timesteps, num_features = data.shape
        num_predictions = num_timesteps - self.model_config.INPUT_CHUNK_LENGTH

        if num_predictions <= 0:
            raise ValueError(f"Not enough data points. Need at least {self.model_config.INPUT_CHUNK_LENGTH + 1} timesteps, got {num_timesteps}")
        
        # This array will store only the predictions for the TARGET ticker
        all_predictions_scaled = np.empty(num_predictions, dtype=np.float32)

        for batch_idx in range(0, num_predictions, self.model_config.EVAL_BATCH_SIZE):
            batch_start = batch_idx
            batch_end = min(batch_idx + self.model_config.EVAL_BATCH_SIZE, num_predictions)
            if batch_idx % (self.model_config.EVAL_BATCH_SIZE * 1000) == 0: # Adjusted frequency for large datasets
                elapsed = time.time() - start_time
                print(f"Processing batch {batch_idx} - {batch_end} / {num_predictions}. Elapsed time: {elapsed:.2f}s")
                
            # Create a batch of multivariate input sequences
            # Each sequence will be (INPUT_CHUNK_LENGTH, num_features)
            indices = np.arange(batch_start, batch_end)[:, None] + np.arange(self.model_config.INPUT_CHUNK_LENGTH)
            batch_input = data[indices] # Shape: (batch_size, INPUT_CHUNK_LENGTH, num_features)

            # predict_future_values returns predictions for the TARGET ticker of shape (batch_size, OUTPUT_CHUNK_LENGTH)
            predictions_for_target_ticker = self.predict_future_values(batch_input)

            # Store the last step of the prediction horizon for each batch item
            all_predictions_scaled[batch_start:batch_end] = predictions_for_target_ticker[:, self.model_config.OUTPUT_CHUNK_LENGTH - 1]

        # Inverse transform the scaled predictions for the target ticker
        # We use the stored min_ and scale_ values for the specific target column
        predicted_mid_prices = (all_predictions_scaled / self.target_column_scale_val) + self.target_column_min_val
        
        return predicted_mid_prices.ravel().tolist()
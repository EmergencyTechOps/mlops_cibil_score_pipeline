import os
import sys
import argparse
import pandas as pd
import xgboost as xgb
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_training():
    logger.info("Starting training...")
    train_dir = os.environ.get("SM_CHANNEL_TRAIN")
    if not train_dir:
        logger.error("ERROR: SM_CHANNEL_TRAIN is not set. Exiting.")
        sys.exit(1)
    logger.info(f"SM_CHANNEL_TRAIN = {train_dir}")
    
    try:
        files = os.listdir(train_dir)
        logger.info(f"Files in SM_CHANNEL_TRAIN: {files}")
    except Exception as e:
        logger.error(f"ERROR listing files in SM_CHANNEL_TRAIN: {e}")
        sys.exit(1)
    
    # Build full paths to the training files.
    X_train_path = os.path.join(train_dir, "X_train.csv")
    y_train_path = os.path.join(train_dir, "y_train.csv")
    
    try:
        logger.info(f"Reading training data from: {X_train_path}")
        X_train = pd.read_csv(X_train_path, header=None)
        logger.info(f"Reading training labels from: {y_train_path}")
        y_train = pd.read_csv(y_train_path, header=None)
    except Exception as e:
        logger.error(f"ERROR reading CSV files: {e}")
        sys.exit(1)
    
    logger.info(f"X_train shape: {X_train.shape}")
    logger.info(f"y_train shape: {y_train.shape}")
    logger.info("Initial X_train dtypes:")
    logger.info(X_train.dtypes)
    
    # Convert any column that is not numeric, boolean, or categorical to a categorical type.
    enable_categorical = False
    for col in X_train.columns:
        if not (pd.api.types.is_numeric_dtype(X_train[col]) or 
                pd.api.types.is_bool_dtype(X_train[col]) or 
                pd.api.types.is_categorical_dtype(X_train[col])):
            logger.info(f"Column {col} is not numeric/boolean/categorical. Converting to category.")
            X_train[col] = X_train[col].astype("category")
            enable_categorical = True
    
    logger.info("After conversion, X_train dtypes:")
    logger.info(X_train.dtypes)
    logger.info(f"enable_categorical = {enable_categorical}")
    
    # Parse hyperparameters (SageMaker passes these via command line)
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--eta", type=float, default=0.1)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample_bytree", type=float, default=0.8)
    parser.add_argument("--eval_metric", type=str, default="rmse")
    parser.add_argument("--num_round", type=int, default=100)
    parser.add_argument("--objective", type=str, default="reg:squarederror")
    args, _ = parser.parse_known_args()
    logger.info(f"Hyperparameters: {args}")
    
    # Create the DMatrix for XGBoost with categorical handling as needed.
    try:
        dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=enable_categorical)
    except Exception as e:
        logger.error(f"ERROR during DMatrix creation: {e}")
        sys.exit(1)
    
    # Set up training parameters.
    params = {
        "max_depth": args.max_depth,
        "eta": args.eta,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "eval_metric": args.eval_metric,
        "objective": args.objective
    }
    logger.info(f"Training parameters: {params}")
    
    # Train the model.
    try:
        bst = xgb.train(params, dtrain, num_boost_round=args.num_round)
    except Exception as e:
        logger.error(f"ERROR during training: {e}")
        sys.exit(1)
    
    # Save the model.
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "xgboost_model.bin")
    try:
        bst.save_model(model_path)
        logger.info(f"Model saved to: {model_path}")
    except Exception as e:
        logger.error(f"ERROR saving model: {e}")
        sys.exit(1)

def main():
    if "SM_CHANNEL_TRAIN" in os.environ:
        logger.info("Detected SM_CHANNEL_TRAIN. Running training inside container...")
        run_training()
    else:
        logger.info("SM_CHANNEL_TRAIN not detected. Launching training job from local environment...")
        launch_training_job()

if __name__ == "__main__":
    main()
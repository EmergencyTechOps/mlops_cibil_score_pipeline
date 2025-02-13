import os
import sys
import argparse
import pandas as pd
import xgboost as xgb

def run_training():
    """
    Runs inside the SageMaker training container.
    Expects SM_CHANNEL_TRAIN (for training data) and SM_MODEL_DIR (for model artifacts) to be set.
    """
    train_dir = os.environ.get("SM_CHANNEL_TRAIN")
    if not train_dir:
        print("ERROR: SM_CHANNEL_TRAIN environment variable is not set. Exiting.")
        sys.exit(1)
    print("SM_CHANNEL_TRAIN =", train_dir)
    
    try:
        files = os.listdir(train_dir)
        print("Files in SM_CHANNEL_TRAIN:", files)
    except Exception as e:
        print("ERROR listing files in SM_CHANNEL_TRAIN:", e)
        sys.exit(1)
    
    # Construct full paths to training files.
    X_train_path = os.path.join(train_dir, "X_train.csv")
    y_train_path = os.path.join(train_dir, "y_train.csv")
    
    try:
        print("Reading training data from:", X_train_path)
        X_train = pd.read_csv(X_train_path, header=None)
        print("Reading training labels from:", y_train_path)
        y_train = pd.read_csv(y_train_path, header=None)
    except Exception as e:
        print("ERROR reading CSV files:", e)
        sys.exit(1)
    
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    
    # Parse hyperparameters passed from SageMaker.
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--eta", type=float, default=0.1)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample_bytree", type=float, default=0.8)
    parser.add_argument("--eval_metric", type=str, default="rmse")
    parser.add_argument("--num_round", type=int, default=100)
    parser.add_argument("--objective", type=str, default="reg:squarederror")
    # Use parse_known_args() to ignore any extra SageMaker arguments.
    args, _ = parser.parse_known_args()
    print("Hyperparameters:", args)
    
    # Train the model.
    try:
        dtrain = xgb.DMatrix(X_train, label=y_train)
        params = {
            "max_depth": args.max_depth,
            "eta": args.eta,
            "subsample": args.subsample,
            "colsample_bytree": args.colsample_bytree,
            "eval_metric": args.eval_metric,
            "objective": args.objective
        }
        print("Training with parameters:", params)
        bst = xgb.train(params, dtrain, num_boost_round=args.num_round)
    except Exception as e:
        print("ERROR during training:", e)
        sys.exit(1)
    
    # Save the model.
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "xgboost_model.bin")
    try:
        bst.save_model(model_path)
        print("Model saved to:", model_path)
    except Exception as e:
        print("ERROR saving model:", e)
        sys.exit(1)

def launch_training_job():
    """
    Runs locally (e.g., in GitHub Actions) to launch a SageMaker training job.
    Uses an existing S3 location for training data if TRAINING_DATA_S3 is set.
    """
    print("Launching SageMaker training job...")
    import sagemaker
    from sagemaker.xgboost.estimator import XGBoost

    role = os.environ.get("SAGEMAKER_ROLE")
    bucket = os.environ.get("S3_BUCKET")
    if not role or not bucket:
        print("ERROR: SAGEMAKER_ROLE or S3_BUCKET environment variable is not set.")
        sys.exit(1)
    
    sagemaker_session = sagemaker.Session()
    
    # If training data is already in S3, use that path.
    training_data_s3 = os.environ.get("TRAINING_DATA_S3")
    if training_data_s3:
        print("Using training data from provided S3 path:", training_data_s3)
    else:
        # Otherwise, default to "s3://{bucket}/data"
        training_data_s3 = f"s3://{bucket}/data"
        print("TRAINING_DATA_S3 not set; defaulting to:", training_data_s3)
    
    # Create the estimator. (This script itself will be used as the entry point.)
    xgb_estimator = XGBoost(
        entry_point="train.py",   # This file is the entry point.
        source_dir=".",           # Repository root.
        framework_version="1.5-1",
        instance_type="ml.c4.2xlarge",
        instance_count=1,
        role=role,
        sagemaker_session=sagemaker_session,
        use_spot_instances=True,
        max_run=3600,
        max_wait=7200
    )
    
    try:
        xgb_estimator.fit({"train": training_data_s3})
        print("SageMaker training job launched successfully!")
    except Exception as e:
        print("ERROR launching SageMaker training job:", e)
        sys.exit(1)

def main():
    # When running inside SageMaker, SM_CHANNEL_TRAIN is set.
    if "SM_CHANNEL_TRAIN" in os.environ:
        print("Detected SM_CHANNEL_TRAIN. Running training inside container...")
        run_training()
    else:
        print("SM_CHANNEL_TRAIN not found. Running locally to launch training job...")
        launch_training_job()

if __name__ == "__main__":
    main()

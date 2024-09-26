from src.components.data_ingestion import DataIngestion

from src.components.data_transformation import DataTransformaion

from src.components.model_trainer import ModelTrainer



def main():
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()   
    data_transformation = DataTransformaion()
    train_data, validation_data = data_transformation.get_data_transformer_object()


    modelTrainer = ModelTrainer()
    model_score = modelTrainer.initiate_model_trainer(train_data, validation_data)

    print("Model Done")


if __name__=="__main__":
    main()
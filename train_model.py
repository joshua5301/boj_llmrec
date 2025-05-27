from boj_llmrec.recommender import Recommender

model_type = 'LightGCN'  # or 'MultiVAE'
recommender = Recommender(data_path='data')
recommender.train_model(model_type=model_type)
recommender.save_model(model_path=f'saved/{model_type}_model.pth', model_type=model_type)
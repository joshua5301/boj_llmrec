import boj_llmrec
import boj_llmrec.recommender

recommender = boj_llmrec.recommender.Recommender(data_path='data')
recommender.train_model()
recommender.save_model(model_path='saved/model.pth')
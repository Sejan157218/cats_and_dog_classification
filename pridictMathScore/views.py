from django.shortcuts import render
from django.views.generic import TemplateView, UpdateView, DetailView
from django.http import HttpResponseRedirect
from django.urls import reverse
import pandas as pd
from src.pipeline.predict_pipeline import PredictPipeline

def Home(request):
   
    return render(request, 'index.html')

class PredictResult(TemplateView):
    def post(self, request):
        custom_data_input_dict ={
        "gender":[request.POST['gender']],
        "race_ethnicity":[request.POST['ethnicity']],
        "parental_level_of_education":[request.POST['parental_level_of_education']],
        "lunch":[request.POST['lunch']],
        "test_preparation_course":[request.POST['test_preparation_course']],
        "reading_score":[float(request.POST['writing_score'])],
        "writing_score":[float(request.POST['reading_score'])],
        }
        data_frame = pd.DataFrame(custom_data_input_dict)

        predicted = PredictPipeline()
        predicted_result = predicted.predict(data_frame)
        print("data_obj", predicted_result)

        return render(request,'index.html', {'predicted_result': int(predicted_result)})

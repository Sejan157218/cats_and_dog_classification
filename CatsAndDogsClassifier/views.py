from django.shortcuts import render
from django.views.generic import TemplateView, UpdateView, DetailView
from django.http import HttpResponseRedirect
from django.urls import reverse
import pandas as pd
from src.pipeline.predict_pipeline import PredictPipeline
from django.shortcuts import render, redirect
from .forms import ImageUploadForm
import PIL
import numpy as np
import tensorflow as tf
from PIL import Image


def Home(request):
   
    return render(request, 'index.html')

class PredictResult(TemplateView):
    def post(self, request):
        form = ImageUploadForm(request.POST, request.FILES)
        file = request.FILES['image']
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # Process the uploaded file here
            uploaded_image = form.cleaned_data['image']
            # You can save the file, process it, or do whatever you need with it
            predicted = PredictPipeline()
            predicted_result = predicted.predict(uploaded_image)
            print("data_obj", predicted_result)
            return render(request,'index.html', {'predicted_result': predicted_result})
        else:
            form = ImageUploadForm()
        return render(request, 'your_template.html', {'form': form})

        # return render(request,'index.html', {'predicted_result': "dogs"})

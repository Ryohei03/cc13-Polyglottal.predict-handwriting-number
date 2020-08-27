import os
from django.shortcuts import render, redirect
from .models import Image
from .forms import ImageForm
from .deep_learning.train_neuralnet import neural
from django.conf import settings

def showall(request):
    images = Image.objects.all()
    result = neural()
    context = {'images':images, 'result':result}
    return render(request, 'app_to_predict/showall.html', context)

def upload(request):
    if request.method == "POST":
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            Image.objects.all().delete()
            if os.path.exists(settings.IMAGE_URL):
                os.remove(settings.IMAGE_URL)
            form.save()
            return redirect('app_to_predict:showall')
    else:
        form = ImageForm()

    context = {'form':form}
    return render(request, 'app_to_predict/upload.html', context)

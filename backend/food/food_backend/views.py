from django.shortcuts import render
from django.http import HttpResponse, request
from .models import Food
from django.shortcuts import redirect
import requests
import json
# from rest_framework import viewsets
# from rest_framework import permissions
# from .serializers import PhotoSerializer

import torch 
import torchvision
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as T
import torchvision.models as models
from torchvision.utils import make_grid
from PIL import Image
import PIL
import re 
import glob
import os

# Create your views here.
def IndexViewSet(request):
    return render(request, 'index.html')

def uploadImage(request):
    pic = request.FILES['image']
    food = Food(img = pic)
    food.save()
    return redirect('/ingredients')

def ingredients(request):
    APP_ID = '0818aa5a'
    APP_KEY = '9c558c2d5c3dffb5476c3be27f2aa197	'
    base_url_ing = "https://api.edamam.com/search?q=chicken&app_id=" + APP_ID + "&app_key=" + APP_KEY
    response = requests.get(base_url_ing)
    response = response.json()
    print(response)
    return render(request, 'ingredients.html')

def model_call(request):
    # image_path = 
    # print(image_path)
    # image_path = image_path.json()
    # image_path = image_path[len(image_path) - 1]
    # image_path = image_path['img']
    # if image_path.startswith("http://localhost:8000/"): 
    #     image_path = image_path.replace("http://localhost:8000/",'') 

    list_of_files = glob.glob('images/*') # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)
    print(latest_file)
    image_path = latest_file
    # image_path = "images/lasag.jpg"
    model_path = "food_backend/results/mod.pth"
    device = 'cpu'
    print("IMAGE PATH : ")
    print(image_path)

    classes = ['apple_pie',
    'baby_back_ribs',
    'baklava',
    'beef_carpaccio',
    'beef_tartare',
    'beet_salad',
    'beignets',
    'bibimbap',
    'bread_pudding',
    'breakfast_burrito',
    'bruschetta',
    'caesar_salad',
    'cannoli',
    'caprese_salad',
    'carrot_cake',
    'ceviche',
    'cheese_plate',
    'cheesecake',
    'chicken_curry',
    'chicken_quesadilla',
    'chicken_wings',
    'chocolate_cake',
    'chocolate_mousse',
    'churros',
    'clam_chowder',
    'club_sandwich',
    'crab_cakes',
    'creme_brulee',
    'croque_madame',
    'cup_cakes',
    'deviled_eggs',
    'donuts',
    'dumplings',
    'edamame',
    'eggs_benedict',
    'escargots',
    'falafel',
    'filet_mignon',
    'fish_and_chips',
    'foie_gras',
    'french_fries',
    'french_onion_soup',
    'french_toast',
    'fried_calamari',
    'fried_rice',
    'frozen_yogurt',
    'garlic_bread',
    'gnocchi',
    'greek_salad',
    'grilled_cheese_sandwich',
    'grilled_salmon',
    'guacamole',
    'gyoza',
    'hamburger',
    'hot_and_sour_soup',
    'hot_dog',
    'huevos_rancheros',
    'hummus',
    'ice_cream',
    'lasagna',
    'lobster_bisque',
    'lobster_roll_sandwich',
    'macaroni_and_cheese',
    'macarons',
    'miso_soup',
    'mussels',
    'nachos',
    'omelette',
    'onion_rings',
    'oysters',
    'pad_thai',
    'paella',
    'pancakes',
    'panna_cotta',
    'peking_duck',
    'pho',
    'pizza',
    'pork_chop',
    'poutine',
    'prime_rib',
    'pulled_pork_sandwich',
    'ramen',
    'ravioli',
    'red_velvet_cake',
    'risotto',
    'samosa',
    'sashimi',
    'scallops',
    'seaweed_salad',
    'shrimp_and_grits',
    'spaghetti_bolognese',
    'spaghetti_carbonara',
    'spring_rolls',
    'steak',
    'strawberry_shortcake',
    'sushi',
    'tacos',
    'takoyaki',
    'tiramisu',
    'tuna_tartare',
    'waffles']

    model = torch.load(model_path, map_location=torch.device(device) )
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()
    # print(model)
        
    test_transforms = T.Compose([
        T.Resize(256),
        T.ToTensor()
    ])

    img = Image.open(image_path)
    image_tensor = test_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    # input = Variable(image_tensor)
    input = image_tensor.to(device)
    output = model(input)
    if device == 'cpu':
        index = output.data.cpu().numpy().argmax()
    else:
        index = output.data.cuda().numpy().argmax()


    print("ANSWER : ",classes[index],"[", index,"]" )

    APP_ID = '0818aa5a'
    APP_KEY = '9c558c2d5c3dffb5476c3be27f2aa197	'
    # food_item = 'cheese_plate'  # replace this to item from model
    food_item = classes[index]
    food_name1 = food_item.replace("_"," ")
    food_item = food_item.replace("_","+")
    base_url_ing = "https://api.edamam.com/search?q=" + food_item + "&app_id=" + APP_ID + "&app_key=" + APP_KEY
    item_details = requests.get(base_url_ing)
    item_details = item_details.json()
    recipes = item_details['hits'][0]['recipe']
    health_labels = recipes['healthLabels']         # categories
    ingredients_all = recipes['ingredientLines']    #ingredients with quantity
    calories = round(recipes['calories'],2)
    context = {"ingredients_all" : ingredients_all, 'item_name': food_item, 'health_label': health_labels, 'calories': calories}
    # context = {}
    return render(request, 'ingredients.html', context)

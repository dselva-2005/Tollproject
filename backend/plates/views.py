import json
import base64
from plates.models import Rfidcard
from django.shortcuts import render
from django.shortcuts import render
from asgiref.sync import async_to_sync
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from channels.layers import get_channel_layer
from django.views.decorators.csrf import csrf_exempt
from plates.models import Reader
from plates.models import DetectedPlates
from django.conf import settings

# Create your views here.
def show_fair(request):
    
    return render(request,'billing.html')


@csrf_exempt
@require_POST
def do_exists(request,car_id):
    print(car_id)
    is_there = Rfidcard.objects.filter(card_number = car_id).exists()
    return JsonResponse({'flag':is_there})

@csrf_exempt
@require_POST
def is_authenticated(request,id):
    api_key = request.headers.get('X-Api-Key')
    device = Reader.objects.get(id=api_key)
    if device:
        data = {}
        exit_distance = device.car_exit.distance_from_entry
        card_key = request.headers.get('X-Card-Key')
        car_card = Rfidcard.objects.get(id = card_key)

        if car_card:
            Found = DetectedPlates.objects.filter(plate_no=car_card.car_number).exists()
            if Found:
                data.update({"distance_travelled":exit_distance})
                data.update({"car_number":car_card.car_number})
                data.update({"total_charge":settings.PRICE_PER_KM*exit_distance})
                data.update({"charge_per_km_cars":settings.PRICE_PER_KM})
                channel_layer = get_channel_layer()
                async_to_sync(channel_layer.group_send)(
                    'chat_lobby',
                    {
                        'type': 'chat_message',
                        'message': json.dumps(data)
                    }
                )
    else: 
        Found = False

    return JsonResponse({'flag':Found})
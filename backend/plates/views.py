from django.shortcuts import render
from django.shortcuts import render
from asgiref.sync import async_to_sync
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from channels.layers import get_channel_layer

# Create your views here.
def show_fair(request):
    
    return render(request,'billing.html')


@require_POST
def my_post_view(request):
    data = request.POST.get('data')

    channel_layer = get_channel_layer()
    async_to_sync(channel_layer.group_send)(
        'chat_lobby',
        {
            'type': 'chat_message',
            'message': data,
        }
    )
    return JsonResponse({'success': True})

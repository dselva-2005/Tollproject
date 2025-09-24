import os
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
from django.core.asgi import get_asgi_application
import plates.routing

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'api.settings')

application = ProtocolTypeRouter({
    "http": get_asgi_application(),  # HTTP requests
    "websocket": AuthMiddlewareStack(
        URLRouter(
            plates.routing.websocket_urlpatterns
        )
    ),
})

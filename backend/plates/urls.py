from . import views
from django.urls import path

app_name = 'plates'

urlpatterns = [
    path('billing',views.show_fair,name='billing_page'),    
]

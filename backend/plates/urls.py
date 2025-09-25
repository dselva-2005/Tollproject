from . import views
from django.urls import path

app_name = 'plates'

urlpatterns = [
    path('billing',views.show_fair,name='billing_page'),    
    path('api/doexists/<slug:car_id>', views.do_exists),
    path('api/is_authenticated/<slug:id>', views.is_authenticated),
]

from django.contrib import admin
from . import models
# Register your models here.

admin.site.register(models.CarExit)
admin.site.register(models.Rfidcard)
admin.site.register(models.DetectedPlates)
admin.site.register(models.Reader)
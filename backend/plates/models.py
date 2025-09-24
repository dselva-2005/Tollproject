from django.db import models
from django.contrib.auth.models import User
import uuid

class DetectedPlates(models.Model):
    plate_no = models.CharField(max_length=20)


class CarExit(models.Model):
    exit_id = models.UUIDField(primary_key=True,default=uuid.uuid4,editable=False)
    distance_from_entry = models.IntegerField()

class Rfidcard(models.Model):
    car_number = models.CharField(max_length=20)
    user = models.OneToOneField(User,on_delete=models.CASCADE)

class Reader(models.Model):
    reader_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    car_exit = models.OneToOneField(CarExit,on_delete=models.CASCADE)
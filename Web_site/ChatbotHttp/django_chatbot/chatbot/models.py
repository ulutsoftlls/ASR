from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
from datetime import datetime
import pytz
# Create your models here.
class Chat(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    message = models.TextField()
    response = models.TextField()
    created_at = models.DateTimeField()

    def save(self, *args, **kwargs):
        # Set the created_at field to the current time in the desired timezone
        if not self.created_at:
            self.created_at = datetime.now(pytz.timezone('Etc/GMT-6'))
        else:
            self.created_at = datetime.now(pytz.timezone('Etc/GMT-6'))
        super(Chat, self).save(*args, **kwargs)

    def __str__(self):
        return f'{self.user.username}'

class Audios(models.Model):
    audio_file = models.FileField(upload_to='', verbose_name="Аудио файл")
    text = models.TextField(blank=True)
    # super_visor = models.CharField(max_length=20)
    # admin = models.CharField(max_length=20)
    # status = models.BooleanField(default=False)
    # is_correct = models.BooleanField(default=False)
    #user_name = models.ForeignKey('Users', on_delete=models.PROTECT, null=True)
    @property
    def sound_display(self):
        if self.audio_file:
            #{self.audio_file.url}
            return mark_safe(f'<audio controls style="width: 300px;" name="media"><source src="{self.audio_file.url}" type="audio/wav"></audio>')
        return ""
    def __str__(self):
        return f" {self.audio_file.url}"

    def get_absolute_url(self):
        return reverse('audio', kwargs={'audio_id': self.pk})

    def save(self, *args, **kwargs):
        super(Audios, self).save(*args, **kwargs)
        return self
    class Meta:
        ordering = ['audio_file']
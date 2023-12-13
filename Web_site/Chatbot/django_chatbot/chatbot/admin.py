from django.contrib import admin
from .models import Chat, Audios
from django.utils.safestring import mark_safe


class ChatAdmin(admin.ModelAdmin):

    list_display = ('id', 'user','get_message' , 'get_response', 'created_at')
    #list_display_links = ('id', 'user')
    search_fields = ('user__username', 'message')
    list_filter = ('user',)
    fields = ('user', 'message', 'response')
    def get_message(self, object):
        if object.message:
            return mark_safe(f"<p style='width: 200px; word-wrap: break-word; font-size: 15px'>{object.message}</p>")
    def get_response(self, object):
        if object.response:
            return mark_safe(f"<p style='width: 500px; word-wrap: break-word; font-size: 15px'>{object.response}</p>")
    get_message.short_description = 'Message'
    get_response.short_description = 'Response'
# Register your models here.

class AudiosAdmin(admin.ModelAdmin):
    #form = MyModel
    list_display = ('id', 'get_audio', 'get_text', 'sound_display')
    list_display_links = ('id', 'get_audio')
    search_fields = ('text', 'audio_file')
    #list_filter = ('status','is_correct','super_visor', 'user_name' ,'admin')
    #list_editable = ('text', )
    fields = ('audio_file', 'text')
    #prepopulated_fields = {"slug": ("audio_file",)}

    def get_audio(self, object):
        if object.audio_file:
            wav_name = str(object.audio_file)
            if len(wav_name)>27:
                wav_name = wav_name[:27] + '...'
            return mark_safe(f"<div style='max-width: 20px table-layout:fixed' >{wav_name}</div>")
    def get_text(self, object):
        if object.text:
            return mark_safe(f"<p style='width: 430px; word-wrap: break-word; font-size: 16px'>{object.text}</p>")
    def check_box(self, object):
        if object.text:
            return mark_safe(f"<p style='width: 180px; word-wrap: break-word;'>{object.text}</p>")
    def sound_display(self, item):
        return item.sound_display

    sound_display.short_description = 'sound'
    sound_display.allow_tags = True
    get_text.short_description = 'Text'
    get_audio.short_description = 'Audio'

admin.site.register(Chat, ChatAdmin)
admin.site.register(Audios, AudiosAdmin)
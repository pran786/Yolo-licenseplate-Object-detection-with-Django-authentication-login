from API import views
from django.urls import path

urlpatterns = [
    path('home', views.index, name = 'index'),
    path('video_feed', views.video_feed,name = 'video_feed'),
    path('stop_video', views.stop_video, name = 'stop_video'),
    path('stream_video', views.stream_video, name = 'stream_video'),
    path('', views.login_view, name='login'),
    path('users_video', views.users_video, name = 'users_video'),
    path('livestream', views.livestream, name = 'livestream'),
    path('logout/', views.logout_view, name='logout'),
]
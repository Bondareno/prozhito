import telebot
from telebot import *
from prozhito_def_files import plot_all_graphs
import io
import matplotlib.pyplot as plt
import traceback

TOKEN='7048581197:AAG6KCjSWBbpyqhA-ZOgnnXkuR_FG3DyCPU'
bot = telebot.TeleBot(TOKEN)

def exit(exitCode):
    print(exitCode)
    print(traceback.format_exc())

@bot.message_handler(commands=['start'])
def start(message):
    bot.send_message(message.chat.id, f'Привет, {message.from_user.first_name}!')
    change_mode(message)
   

def change_mode(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    btn1 = types.KeyboardButton("🔍 Найти")
    btn2 = types.KeyboardButton("🛈 Инструкция")
    markup.add(btn1, btn2)
    
    bot.send_message(message.chat.id, f'Чем могу помочь?', reply_markup=markup)
    bot.register_next_step_handler(message, mode_router)
    
    
def mode_router(message):
    if message.text == '🔍 Найти':
        search(message)
        
    elif message.text == '🛈 Инструкция':
        try:        
            with open('source/instruction.pdf', 'rb') as file:
                bot.send_document(message.chat.id, file)    
            change_mode(message)
            
        except Exception as e:
            tb = traceback.format_exc()
            print(tb)
            exit("Failed to convert name" + str(e))
            bot.send_message(message.chat.id, 'Файл пока недоступен')
            change_mode(message)
        
    else:
        bot.send_message(message.chat.id, 'Неверный ввод. Выберите один из вариантов(кнопок).')
        change_mode(message)


def search(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True).add(types.KeyboardButton("🔙 Назад"))
    bot.send_message(message.chat.id, 'Отправьте мне id автора', reply_markup=markup)
    bot.register_next_step_handler(message, step_1)
    
def step_1(message):
    if message.text == '🔙 Назад':
        change_mode(message)
            
    else:
        if message.text.isdigit():
            try:
                bot.send_message(message.chat.id, 'Ищу')
                
                fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(18.5, 14), layout="constrained")
                
                fig, messages = plot_all_graphs(fig , axs, message.text)
                
                if fig == None and messages ==None:
                    bot.send_message(message.chat.id, 'Нет такого id')
                    search(message)
                    return None
                    
                # Сохранение изображения в поток байтов
                buf = io.BytesIO()
                fig.savefig(buf, format='png')
                buf.seek(0)

                for text in messages:
                    bot.send_message(message.chat.id, text)
                    
                # Отправка изображения
                bot.send_photo(message.chat.id, buf)

                # Закрытие изображения
                plt.clf()
                plt.close('all')
                change_mode(message)

            except Exception as e:                  
                    tb = traceback.format_exc()
                    print(tb)
                    exit("Failed to convert name" + str(e))
                    bot.send_message(message.chat.id, 'Непредвиденная ошибка')
                    search(message)
        
        else:
            bot.send_message(message.chat.id, 'Нужно ввести число')
            bot.register_next_step_handler(message, step_1)
        
bot.infinity_polling()

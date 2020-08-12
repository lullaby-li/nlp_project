# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 16:59:12 2019

@author: yihua
"""
from flask import Flask, url_for, render_template, flash, redirect, request, session
from flask_wtf import FlaskForm
from wtforms import StringField, TextAreaField, SubmitField
from wtforms.validators import DataRequired, Length, Email
from wtforms import *
from urllib import parse
import random

from _4_end_to_end import * 

bad_response = [
"哎呀，你好像没有输入正文哦！｡:.ﾟヽ(｡◕‿◕｡)ﾉﾟ.:｡+ﾟ", 
"逗我呢，没有正文怎么让俺写摘要？(╯‵□′)╯︵┻━┻",
"亲，你忘记输入正文了哦！╭(′▽`)╭(′▽`)╯",
"emmmm，不输入正文的话就算是人工智能也无力呢！(๑•́ ₃•̀๑)",
"请输入正文！(;¬_¬)",
"警察蜀黍，就是这个人！๑乛◡乛๑"
]


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('inx.html')

@app.route('/testPost', methods=['post'])
def testPost():
    a = request.get_data()
    a = parse.parse_qs(a.decode())
#    json_data = json.loads(a)
#    response = json_data.get("title")
    try:
        content = a['content'][0]
    except KeyError:
        content = ""
    try:
        title = a['title'][0]
    except KeyError:
        title = ""

    if not content: return random.choice(bad_response)
    return getAbstract(content=content, title=title)


if __name__ == "__main__":
    app.run(debug=True)
#    with app.test_request_context():
#        print(url_for("login", name='aa'))
#        print(url_for("good", name='aa'))
        
#a = 'title=%E9%98%BF%E6%96%AF%E8%92%82%E8%8A%AC%2C%E4%BD%86%E6%98%AF&content=%E6%92%92%E6%97%A6'
#parse.parse_qs(a)

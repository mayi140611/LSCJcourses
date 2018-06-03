# -*- encoding=utf-8 -*-

import web
import json
        
urls = (
    '/hello/(.*)', 'hello',
    '/checkphone/(.+)', 'CheckPhone',
    '/showfile/(.+)', 'ShowFile',
    '/showtips/(.+)', 'ShowTips',
)
app = web.application(urls, globals())

class hello:        
    def GET(self, name):
        if not name: 
            name = 'World'
        return 'Hello, ' + name + '!'

class CheckPhone:
    def GET(self, phone):
        web.header('content-type','text/json')
        if not phone:
            return json.dumps({"HEAD": "error", "BODY":"参数错误"})
        if phone == '13000000000':
            return json.dumps({"HEAD": "error", "BODY":"手机号已存在"})
        return json.dumps({"HEAD": "ok", "BODY":""})

class ShowTips:
    def GET(self, phone):
        web.header('content-type','text/json')
        web.header('Access-Control-Allow-Origin','*')
        return json.dumps({"HEAD": "error", "BODY":["宋小宝","郭德纲","于谦","贾玲","岳云鹏"]})
class ShowFile:
    def GET(self, filename):
        f = open('./%s'%filename, 'r')
        html = f.read()
        f.close()
        return html


if __name__ == "__main__":
    app.run()

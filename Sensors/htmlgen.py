#!/usr/bin/env python
import bottle

# The HTML template to use
template = """
    <html>
    <head>
    <meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
    </head>
    <body>
    % # items contains what it was passed in the bottle.template() function
    <h1>Varian XGS600</h1>
    <table>
    <tr>
      <td>IMG1:</td>
      <td>{{items['p1_xgs']}}</td>
      <td>Torr</td>
    </tr>
    <tr>
      <td>CNV1:</td>
      <td>{{items['p2_xgs']}}</td>
      <td>Torr</td>
    </tr>
    </table> 
        
    <h1>LakeShore 336</h1>
    <table>
    <tr>
      <td>Sensor A:</td>
      <td>{{items['t1_336']}}</td>
      <td>K</td>
    </tr>
    <tr>
      <td>Sensor B:</td>
      <td>{{items['t2_336']}}</td>
      <td>K</td>
    </tr>
    <tr>
      <td>Sensor C:</td>
      <td>{{items['t3_336']}}</td>
      <td>K</td>
    </tr>
    <tr>
      <td>Sensor D:</td>
      <td>{{items['t4_336']}}</td>
      <td>K</td>
    </tr>
    </table> 

    <h1>LakeShore 218</h1>
    <table>
    <tr>
      <td>Sensor 1:</td>
      <td>{{items['t1_218']}}</td>
      <td>K</td>
    </tr>
    <tr>
      <td>Sensor 2:</td>
      <td>{{items['t2_218']}}</td>
      <td>K</td>
    </tr>
    <tr>
      <td>Sensor 3:</td>
      <td>{{items['t3_218']}}</td>
      <td>K</td>
    </tr>
    <tr>
      <td>Sensor 4:</td>
      <td>{{items['t4_218']}}</td>
      <td>K</td>
    </tr>
    <tr>
      <td>Sensor 5:</td>
      <td>{{items['t5_218']}}</td>
      <td>K</td>
    </tr>
    <tr>
      <td>Sensor 6:</td>
      <td>{{items['t6_218']}}</td>
      <td>K</td>
    </tr>
    <tr>
      <td>Sensor 7:</td>
      <td>{{items['t7_218']}}</td>
      <td>K</td>
    </tr>
    <tr>
      <td>Sensor 8:</td>
      <td>{{items['t8_218']}}</td>
      <td>K</td>
    </tr>
    </table> 

    </body>
    </html>"""

def pageGen(sensors):
    """Take sensor outputs and generate web page"""
    # to use the template string above
    html = bottle.template(template, items=sensors)

    # write html page
    with open("iconn_sensors.html", "wb") as f:
        f.write(html.encode('utf8'))

if __name__ == '__main__':
# dictionary of sensors
    sensors = {"p1_xgs":0.0001, "p2_xgs":0.0000002, "t1_336":122.0, "t2_336":1.1, "t3_336":1.2, "t4_336":1.3, 
            "t1_218":2.0, "t2_218":122.1, "t3_218":2.2, "t4_218":2.3, "t5_218":2.4, "t6_218":2.5, "t7_218":2.6, "t8_218":2.7}

    pageGen(sensors)

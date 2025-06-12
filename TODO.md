- Mejorar funcion de rewards
- Entrenar red Q en entorno simulado con 1000 episodios sin render y guardar pesos:
   `uv run python src/doro/utils/train_agent.py --episodes 1000`
- Entrenar LSTM de nuevo con el nuevo modulo `LSTMFeatureExtractor`
- Rellenar execute_action usando control.Control: (main.py del cliente libreria Freenove)
  
  

  
  
  
   if (event.key() == Qt.Key_W):
            if not (event.isAutoRepeat()) and self.Key_W == True:
                print("release W")
                self.Key_W = False
                self.move_point = [325, 635]
                self.move()

                self.move_point = [325, 535]
                self.move()



try:
            x = self.map((self.move_point[0]-325),0,100,0,35)
            y = self.map((635 - self.move_point[1]),0,100,0,35)
            if self.action_flag == 1:
                angle = 0
            else:
                if x!=0 or y!=0:
                    angle=math.degrees(math.atan2(x,y))

                    if angle < -90 and angle >= -180:
                        angle=angle+360
                    if angle >= -90 and angle <=90:
                        angle = self.map(angle, -90, 90, -10, 10)
                    else:
                        angle = self.map(angle, 270, 90, 10, -10)
                else:
                    angle=0
            speed=self.client.move_speed
            command = cmd.CMD_MOVE+ "#"+str(self.gait_flag)+"#"+str(round(x))+"#"+str(round(y))\
                      +"#"+str(speed)+"#"+str(round(angle)) +'\n'
            print(command)
            self.client.send_data(command)
        except Exception as e:
            print(e)
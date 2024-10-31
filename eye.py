import cv2
import mediapipe as mp
import utils

def main(flip=True):
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils

    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 將畫面上下顛倒
        if flip:
            frame = cv2.flip(frame, 0)  # 0表示上下翻轉

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)

                # 獲取面部特徵點
                left_eye = face_landmarks.landmark[159]
                right_eye = face_landmarks.landmark[386]
                nose = face_landmarks.landmark[1]
                mouth_left = face_landmarks.landmark[61]
                mouth_right = face_landmarks.landmark[291]

                # 可以在這裡加入判斷正反面的邏輯

                # 計算眼睛的開合狀態
                left_eye_height = face_landmarks.landmark[145].y - left_eye.y
                right_eye_height = face_landmarks.landmark[374].y - right_eye.y

                if left_eye_height < 0.01 and right_eye_height < 0.01:
                    cv2.putText(frame, "Eyes Closed", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    utils.light_flash()
                else:
                    cv2.putText(frame, "Eyes Open", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    utils.light_off()

        cv2.imshow('Eye Blink Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    # main(flip=False)
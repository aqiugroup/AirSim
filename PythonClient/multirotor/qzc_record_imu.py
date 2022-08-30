# coding=utf-8
import airsim
import time


def parseIMU(imu_data):
    angular_velocity = imu_data.angular_velocity
    linear_acceleration = imu_data.linear_acceleration
    orientation = imu_data.orientation
    time_stamp = imu_data.time_stamp

    # 参考EuRoC IMU数据格式
    data_item = [str(time_stamp),
                 str(angular_velocity.x_val),
                 str(angular_velocity.y_val),
                 str(angular_velocity.z_val),
                 str(linear_acceleration.x_val),
                 str(linear_acceleration.y_val),
                 str(linear_acceleration.z_val)]
    return data_item


if __name__ == '__main__':
    # 连接到AirSim模拟器
    client = airsim.MultirotorClient()
    client.confirmConnection()

    cur_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    fout = open("/Users/aqiu/Documents/AirSim/"+cur_time + "_IMU.csv", "w")
    fout.write("# timestamp(ns)、"
               "gyro_x(rad/s)、gyro_y(rad/s)、gyro_z(rad/s)、"
               "accel_x(m/s^2)、accel_y(m/s^2)、accel_z(m/s^2)\n")

    print("Recording IMU data ...\nPress Ctrl + C to stop.")
    last_timestamp = 0
    # 循环读取数据
    while True:
        # 通过getImuData()函数即可获得IMU观测
        # 返回结果由角速度、线加速度、朝向(四元数表示)、时间戳(纳秒)构成
        imu_data = client.getImuData()
        cur_time_stamp = imu_data.time_stamp

        if cur_time_stamp != last_timestamp:
            data_item = parseIMU(imu_data)
            fout.write(data_item[0] + "," +
                       data_item[1] + "," +
                       data_item[2] + "," +
                       data_item[3] + "," +
                       data_item[4] + "," +
                       data_item[5] + "," +
                       data_item[6] + "\n")
            last_timestamp = cur_time_stamp

    fout.close()

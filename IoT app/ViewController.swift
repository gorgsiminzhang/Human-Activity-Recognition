//
//  ViewController.swift
//  HAR
//
//  Created by Simin Zhang on 2021-02-26.
//  Copyright © 2021 Simin Zhang. All rights reserved.
//

import UIKit
import CoreMotion
import AzureIoTHubClient
import Foundation

class ViewController: UIViewController{
    
    //Mark:Properties
   
    @IBOutlet weak var Gyrox: UILabel!
    @IBOutlet weak var Gyroy: UILabel!
    @IBOutlet weak var Gyroz: UILabel!
    @IBOutlet weak var Accx: UILabel!
    @IBOutlet weak var Accy: UILabel!
    @IBOutlet weak var Accz: UILabel!
    
    var motion = CMMotionManager()
    
    //Put you connection string here
    private let connectionString = "HostName=SZIoT.azure-devices.net;DeviceId=myiOSdevice;SharedAccessKey=jbMvcCtiBx3s4RxphYbSqYIXSvwxgiS6QSrK1JcqpYo="
    
    // Select your protocol of choice: MQTT_Protocol, AMQP_Protocol or HTTP_Protocol
    // Note: HTTP_Protocol is not currently supported
    private let iotProtocol: IOTHUB_CLIENT_TRANSPORT_PROVIDER = MQTT_Protocol
    
    // IoT hub handle
    private var iotHubClientHandle: IOTHUB_CLIENT_LL_HANDLE!;
    
    
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view, typically from a nib.
    }

    func MyGyro(){
        motion.gyroUpdateInterval = 0.5
        motion.startGyroUpdates(to: OperationQueue.current!){(data,error) in
            if let trueData = data{
                self.Gyrox.text = "\(trueData.rotationRate.x)"
                self.Gyroy.text = "\(trueData.rotationRate.y)"
                self.Gyroz.text = "\(trueData.rotationRate.z)"
            }
            
        }
    }
    
    func MyAcc(){
        
        motion.accelerometerUpdateInterval = 0.5
        motion.startAccelerometerUpdates(to: OperationQueue.current!){(data,error) in
            if let trueData = data{
                self.Accx.text = "\(trueData.acceleration.x)"
                self.Accy.text = "\(trueData.acceleration.y)"
                self.Accz.text = "\(trueData.acceleration.z)"
            }
            
        }
        
    }

}


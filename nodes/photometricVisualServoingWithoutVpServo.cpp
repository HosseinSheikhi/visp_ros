/****************************************************************************
 *
 * ViSP, open source Visual Servoing Platform software.
 * Copyright (C) 2005 - 2019 by Inria. All rights reserved.
 *
 * This software is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 * See the file LICENSE.txt at the root directory of this source
 * distribution for additional information about the GNU GPL.
 *
 * For using ViSP with software that can not be combined with the GNU
 * GPL, please contact Inria about acquiring a ViSP Professional
 * Edition License.
 *
 * See http://visp.inria.fr for more information.
 *
 * This software was developed at:
 * Inria Rennes - Bretagne Atlantique
 * Campus Universitaire de Beaulieu
 * 35042 Rennes Cedex
 * France
 *
 * If you have questions regarding the use of this file, please contact
 * Inria at visp@inria.fr
 *
 * This file is provided AS IS with NO WARRANTY OF ANY KIND, INCLUDING THE
 * WARRANTY OF DESIGN, MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
 *
 * Authors:
 * Eric Marchand
 * Christophe Collewet
 *
 *****************************************************************************/

/*!
  \example photometricVisualServoingWithoutVpServo.cpp

  Implemented from \cite Collewet08c.
*/

#include <visp3/core/vpDebug.h>

#include <visp3/core/vpImage.h>
#include <visp3/core/vpImageTools.h>
#include <visp3/io/vpImageIo.h>

#include <visp3/core/vpCameraParameters.h>
#include <visp3/core/vpTime.h>
#include <visp3/robot/vpSimulatorCamera.h>

#include <visp3/core/vpHomogeneousMatrix.h>
#include <visp3/core/vpMath.h>
#include <visp3/gui/vpDisplayD3D.h>
#include <visp3/gui/vpDisplayGDI.h>
#include <visp3/gui/vpDisplayGTK.h>
#include <visp3/gui/vpDisplayOpenCV.h>
#include <visp3/gui/vpDisplayX.h>

#include <visp3/io/vpParseArgv.h>
#include <visp3/visual_features/vpFeatureLuminance.h>

#include <stdlib.h>
#include <visp3/robot/vpImageSimulator.h>

#include <visp3/core/vpIoTools.h>
#include <visp3/io/vpParseArgv.h>

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>

#include <boost/thread/mutex.hpp>
#include <image_transport/image_transport.h>
#include <ros/ros.h>
#include <std_msgs/Int8.h>
#include <std_msgs/String.h>
#include <std_srvs/Empty.h>
#include <std_srvs/EmptyRequest.h>
#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/Twist.h>

#include <visp3/blob/vpDot2.h>
#include <visp3/core/vpImageConvert.h>
#include <visp3/core/vpPixelMeterConversion.h>
#include <visp3/gui/vpDisplayGDI.h>
#include <visp3/gui/vpDisplayOpenCV.h>
#include <visp3/gui/vpDisplayX.h>
#include <visp3/vision/vpPose.h>
#define Z 1


namespace
{
class Photometric
{

  ros::NodeHandle m_nh;
  image_transport::ImageTransport m_it;
  image_transport::Subscriber m_image_sub, m_image_sub_goal, m_image_sub_home;
  ros::ServiceClient reset_req;
  bool needs_reset = true;
  bool reached_goal, reached_home, dvs_init, image_update;
  ros::Publisher velocity_publisher;
  geometry_msgs::Twist camvel_msg;
  unsigned int m_queue_size;
  boost::mutex m_lock;
#ifdef VISP_HAVE_X11
  vpDisplayX *m_display;
#elif defined( VISP_HAVE_GDI )
  vpDisplayGDI *m_display;
#elif defined( VISP_HAVE_OPENCV )
  vpDisplayOpenCV *m_display;
#endif
  vpImage<unsigned char > I, current;  
  vpImage<unsigned char> Id;
  vpImage<unsigned char> Idiff;
  vpFeatureLuminance sI, sId;
  // Matrice d'interaction, Hessien, erreur,...
  vpMatrix Lsd;      // matrice d'interaction a la position desiree
  vpMatrix Hsd;      // hessien a la position desiree
  vpMatrix H;        // Hessien utilise pour le levenberg-Marquartd
  vpColVector error; // Erreur I-I*
  vpCameraParameters cam;

  // ------------------------------------------------------
  // Control law
  double lambda=30; // gain
  vpColVector e;
  vpColVector v; // camera velocity send to the robot

  // ----------------------------------------------------------
  // Minimisation

  double mu=0.05; // mu = 0 : Gauss Newton ; mu != 0  : LM
  double lambdaGN=30;

    // ----------------------------------------------------------
  int iter = 1;
  int iterGN = 510; // swicth to Gauss Newton after iterGN iterations

  double normeError = 0;
  int opt_niter = 400;
  double min_error = 1000000000;

  unsigned int n = 6;
  vpMatrix diagHsd;
  int total_tasks = 0;
  int succed_tasks = 0;
public:
  Photometric()
    : m_it( m_nh )
    , m_image_sub()
    , reached_goal(false)
    , reached_home(false)
    , dvs_init(false)
    , image_update(false)
    , m_queue_size( 1 )
    , m_lock()
    , m_display( NULL )
    , cam(320, 240, 160, 120)
    , diagHsd(n,n)
  {
    m_image_sub_goal = m_it.subscribe( "isaac/ros/raw_image_goal", m_queue_size, &Photometric::callback_goal, this );
    m_image_sub_home = m_it.subscribe( "isaac/ros/raw_image_home", m_queue_size, &Photometric::callback_home, this );
    m_image_sub = m_it.subscribe( "isaac/ros/raw_image", m_queue_size, &Photometric::callback, this );
    velocity_publisher = m_nh.advertise<geometry_msgs::Twist>("/dvs/velocity", 100);
    reset_req = m_nh.serviceClient<std_srvs::Empty>("reset");
    
    


  }

  void init_dvs(){
    // current visual feature built from the image
    // (actually, this is the image...)
    
    sI.init(I.getHeight(), I.getWidth(), Z);
    sI.setCameraParameters(cam);
    sI.buildFrom(I);

    // desired visual feature built from the image
    sId.init(Id.getHeight(), Id.getWidth(), Z);
    sId.setCameraParameters(cam);
    sId.buildFrom(Id);

    // init Idiff and calculate the difference
    Idiff = I;
    vpImageTools::imageDifference(I, Id, Idiff);

    // ------------------------------------------------------
    // Visual feature, interaction matrix, error
    // s, Ls, Lsd, Lt, Lp, etc
    // ------------------------------------------------------

    // Compute the interaction matrix
    // link the variation of image intensity to camera motion

    // here it is computed at the desired position
    sId.interaction(Lsd);

    // Compute the Hessian H = L^TL
    Hsd = Lsd.AtA();

    // Compute the Hessian diagonal for the Levenberg-Marquartd
    // optimization process
    
    diagHsd.eye(n);
    for (unsigned int i = 0; i < n; i++)
      diagHsd[i][i] = Hsd[i][i];

    dvs_init = true;
  }
  
  void spin()
  {
    ros::Rate loop_rate( 10 );

    while ( m_nh.ok() )
    {
      // should call DVS func here
      ros::spinOnce();
      
      if(needs_reset){
        std_srvs::Empty srv;
        if (reset_req.call(srv))
        {
          needs_reset = false;
          reached_goal = false;
          reached_home = false;
          dvs_init = false;
          image_update = false;
          iter = 1;
          min_error = 1000000000;
          
          ROS_INFO("responce received");
        }
        else
        {
          ROS_ERROR("Failed to call service reset");
        }
      }
      else{

        if(reached_home and reached_goal and !dvs_init){
            init_dvs();
        }
        else if(image_update and reached_home and reached_goal and dvs_init){ // main loop
          if( iter>10 and (min_error < 20000000 || iter > opt_niter)){
            total_tasks++;
            if(min_error<20000000)
              succed_tasks++;
            ROS_INFO("Total_tasks: %d   Succed_task:%d", total_tasks, succed_tasks);
            needs_reset = true;
          }
          std::cout << "--------------------------------------------" << iter++ << std::endl;


          vpImageTools::imageDifference(I, Id, Idiff);
          if ( m_display == NULL )
          {
            init_display();
          }
          vpDisplay::display( Idiff );
          vpDisplay::flush( Idiff );

          // Compute current visual feature
          sI.buildFrom(I);

          // compute current error
          sI.error(sId, error);

          normeError = (error.sumSquare());
          std::cout << "|e| " << normeError << std::endl;
          min_error = std::min(normeError, min_error);
          std::cout << "|min e| " << min_error << std::endl;

          // ---------- Levenberg Marquardt method --------------
          {
            if (iter > iterGN) {
              mu = 0.001;
              lambda = lambdaGN;
            }

            // Compute the levenberg Marquartd term
            {
              H = ((mu * diagHsd) + Hsd).inverseByLU();
            }
            //	compute the control law
            e = H * Lsd.t() * error;

            v = -lambda * e;
          }

          std::cout << "lambda = " << lambda << "  mu = " << mu;
          std::cout << " |Tc| = " << sqrt(v.sumSquare()) << std::endl;

          // send the robot velocity
          // v.cppPrint(std::cout, "velocity");
          
          camvel_msg.linear.x  = v[0];
          camvel_msg.linear.y  = v[1];
          camvel_msg.linear.z  = v[2];
          camvel_msg.angular.x = v[3];
          camvel_msg.angular.y = v[4];
          camvel_msg.angular.z = v[5];
          velocity_publisher.publish( camvel_msg );
          image_update = false;
        }
      }  
      // else if(!image_update and camvel_msg.linear.x!=0)
      //   velocity_publisher.publish( camvel_msg );
      loop_rate.sleep();
    }
  }


  void init_display()
  {
    #ifdef VISP_HAVE_X11
        m_display = new vpDisplayX;
    #elif VISP_HAVE_GDI
        m_display = new vpDisplayGDI;
    #elif VISP_HAVE_OPENCV
        m_display = new vpDisplayOpenCV;
    #endif
    if ( m_display )
    {
      std::cout << "Image size: " << Idiff.getWidth() << " x " << Idiff.getHeight() << std::endl;
      m_display->init( Idiff, 120, Idiff.getHeight() + 20, "error" );
    }
  }


  void callback( const sensor_msgs::ImageConstPtr &msg )
  {
    if(msg->header.frame_id != std::to_string(iter)){
      std::cout<<"iter not matched  "<<msg->header.frame_id <<"  "<<iter<<std::endl;
      return;
    }
    // std::cout<<msg->header.frame_id<<std::endl;
    // std::cout<<"main called\n";

    // boost::mutex::scoped_lock( m_lock );    
    cv::Mat cv_frame = cv_bridge::toCvShare( msg, "bgr8" )->image;
    vpImageConvert::convert( cv_frame, I );
    
    image_update = true;
  }

  void callback_goal( const sensor_msgs::ImageConstPtr &msg )
  {
        // boost::mutex::scoped_lock( m_lock );   

    std::cout<<"goal image called\n";
    if(reached_goal)return; 
    cv::Mat cv_frame = cv_bridge::toCvShare( msg, "bgr8" )->image;
    vpImageConvert::convert( cv_frame, Id );
    if(Id.getHeight()>0){
      reached_goal = true;
      std::cout<<"goal image received\n";
    }
  }

  void callback_home( const sensor_msgs::ImageConstPtr &msg )
  {
        // boost::mutex::scoped_lock( m_lock );   

    std::cout<<"home image called\n";
    if(reached_home) return;    
    cv::Mat cv_frame = cv_bridge::toCvShare( msg, "bgr8" )->image;
    vpImageConvert::convert( cv_frame, I );
    if(I.getHeight()>0){
      reached_home = true;
      std::cout<<"home image received\n";
      image_update = true;
    }

  }

};
} // namespace

int
main( int argc, char **argv )
{
  ros::init( argc, argv, "photometric" );
  Photometric photometric;
  
  // position the robot at home and get I
  photometric.spin();
  return 0;
}

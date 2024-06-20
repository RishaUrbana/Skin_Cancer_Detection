<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document</title>
    <link rel="stylesheet" href="css/all.min.css">
    <link rel="stylesheet" href="css/fontawesome.min.css">
    <link rel="stylesheet" href="css/bootstrap.min.css">
  <link rel="stylesheet" href="generator.css">
  <link rel="stylesheet" href="css/animate.min.css">
</head>
<body>
     <!--Start of a navbar-->
    <nav class="navbar navbar-expand-lg navbar-light bg-color sticky-top">
      <a class="navbar-brand" href="#"><img src="images/Blue Minimalist Medical Logo1.png" class="img-fluid logo-image" alt="Responsive image"></a>
      <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarSupportedContent" aria-controls="navbarSupportedContent" aria-expanded="false" aria-label="Toggle navigation">
        <span class="navbar-toggler-icon"></span>
      </button>
    
      <div class="collapse navbar-collapse" id="navbarSupportedContent">
        <ul class="navbar-nav mr-auto nav-margin">
          <li class="nav-item ">
            <a class="nav-link nav-color ac-color animate__animated animate__zoomIn wow zoomIn" href="index.php">Home <span class="sr-only">(current)</span></a>
          </li>
          <li class="nav-item">
            <a class="nav-link nav-color animate__animated animate__zoomIn wow zoomIn " href="index.php">About</a>
          </li>
          <li class="nav-item dropdown">
            <a class="nav-link dropdown-toggle nav-color animate__animated animate__zoomIn wow zoomIn" href="#" id="navbarDropdown" role="button" data-toggle="dropdown" aria-haspopup="true" aria-expanded="false">
              Skin Cancer
            </a>
            <div class="dropdown-menu" aria-labelledby="navbarDropdown">
              <a class="dropdown-item" href="index.php">Nevus </a>
              <a class="dropdown-item" href="index.php">pigmented benign keratosis</a>
              <a class="dropdown-item" href="index.php">Melanoma</a>
            </div>
          </li>
          <li class="nav-item active">
            <a class="nav-link nav-color animate__animated animate__zoomIn wow zoomIn" href="login.php">Detection</a>
          </li>

          <div class="btn"><a href="logindoctor.html" class="a1 log wow animate__zoomIn" > LOGIN</a></div> 
            <div class="btn"><a href="regi.html" class="a1 wow animate__zoomIn"> SIGNUP</a ></div> 
          
        </ul>

      </div>
    </nav>
   <!--end of a navbar-->

      <!--Start of a background-->
  <div class="bg">
   <div class="container">
       <div class="row">
           <div class="col-sm-12">
            <div class="jumbotron box animate__animated animate__zoomIn wow zoomIn">
             <h2 class="header animate__animated animate__zoomIn wow zoomIn">DETECT YOUR SKIN CANCER USING OUR ONLINE SKIN CANCER DETECTOR</h2>
            
             <form>
                 <div class="container">
                     <div class="row">
                         <div class="col-sm-12 mt-5">
                          <label for="inputState">Capture and Upload a picture:</label>
                          <div class="custom-file mt-3">
                           
                   
                            <input type="file" class="custom-file-input ic" id="customFile" name="image1">
                       
                          <label class="custom-file-label ic" for="customFile"></label>
                          
                              </div>
    
                         </div>
              </div>
              </div>
               
              </form>

              <button type="submit" class="btn my_btn_1 mt-5 px-5">GENERATE verdict</button>   
            
            
           
           
        </div>
           </div>
       </div>
   </div>



  <!--start of footer section-->
      
  <footer class="text-center">
    <p class="f-text pt-5 text-white animate__animated animate__zoomIn wow zoomIn"><i class="far fa-copyright"></i> COPYRIGHTS 2024 <span class="logo-text">ONLINE SKIN CANCER DETECTOR </span> THEME ALL RIGHTS RESERVED</p>
   </footer>



    <!--start of footer section-->
  </div>
 <!--end of a background-->



    <script src="js/jquery-3.5.1.js"></script>
    <script src="js/popper.min.js"></script>
    <script src="js/bootstrap.min.js"></script>
    <script src="js/wow.js"></script>
    <script>
       new WOW().init();
    </script>

</body>
</html>

### <ins>Docker Overview:</ins>

1. Docker is an open-source containerized platform designed to create, deploy and run application
2. Docker uses container on Host OS to run applications, it allows application to use same Linux kernel as a system on the host computer, rather than creating a whole virtual OS
3. We can install docker on any OS  but docker engine runs natively on Linux distribution
4. Docker written in Go lanuage
5. Docker is tool that performs OS level virtualization, also known as containerization
6. Before docker many users faces the problem that a particular code is running in developer system but not in tester system
7. Docker is a set of Platform as a service that uses OS level virtualization where as VMware uses Hardware level virtualization

### <ins>Docker Advantages:</ins>

1. No pre-allocation of RAM
2. CI efficiency - Docker enables you to build a container image and use that same image across every step of the deployment process
3. Less cost (because of OS level virtualization)
4. Light-weight
5. Run on physical hardware, vitual hardware and on cloud
6. Re-use the image
7. Take less time to create container

### <ins>Docker Limitations:</ins>

1. Docker is not good solution for application that requires rich UI
2. Difficult to manage large amount of containers
3. Docker doesn't provide cross-platform compatabality means if a application is designed to run in a docker container on windows then it can't run on linux or vice-versa
4. Docker is suitable when the development OS and testing OS are same, if the OS is different then we should use VM
5. No solution for data recovery and backup


### <ins>Docker Volume:</ins>

1. Volume is simply a directory inside a container
2. Firstly, we have to decalre directory as a volume and then share the volume
3. Even if we stop container, still we can access volume
4. Volume will be created in one container
5. Can declare a directory as volume only while creaating container
6. Can't create volume from existing containers
7. Can share volume across any number of containers
8. volume will not be included when you update an image (meaning when you create a image using existing container and then use that extracted image to create new container then the volume existed earlier won't act as a volume in new container)
9. Can map volume in 2 ways-
	- Container to Container
	- Host to Container
	
### <ins>Benefits of creating a volume:</ins>

1. Decoupling container from storage
2. Share volume among different containers
3. Attach volume to containers
4. Deleting a container won't delete the volume
---

#include <GL/glut.h>
#include <math.h>
#include <stdlib.h>

float mercury=0,venus=0,earth=0,mars=0;
float moon=0,satellite=0;
float saturn=0;

float camX=0,camY=25,camZ=90;
float yaw=0;

const int STAR_COUNT=1000;
float stars[STAR_COUNT][3];

const int ASTEROID_COUNT=200;
float asteroidAngle[ASTEROID_COUNT];
float asteroidRadius[ASTEROID_COUNT];

float cometX=-200;

void initStars()
{
    for(int i=0;i<STAR_COUNT;i++)
    {
        stars[i][0]=(rand()%600)-300;
        stars[i][1]=(rand()%600)-300;
        stars[i][2]=(rand()%600)-300;
    }
}

void initAsteroids()
{
    for(int i=0;i<ASTEROID_COUNT;i++)
    {
        asteroidAngle[i]=rand()%360;
        asteroidRadius[i]=26+rand()%6;
    }
}

void drawStars()
{
    glDisable(GL_LIGHTING);

    glPointSize(2);

    glBegin(GL_POINTS);

    for(int i=0;i<STAR_COUNT;i++)
    {
        glColor3f(1,1,1);
        glVertex3f(stars[i][0],stars[i][1],stars[i][2]);
    }

    glEnd();

    glEnable(GL_LIGHTING);
}

void drawSun()
{
    GLfloat emission[]={1.0,0.6,0.0,1.0};
    GLfloat noEmission[]={0,0,0,1};

    glMaterialfv(GL_FRONT,GL_EMISSION,emission);

    glColor3f(1,0.8,0);

    glutSolidSphere(4,60,60);

    glMaterialfv(GL_FRONT,GL_EMISSION,noEmission);

    glDisable(GL_LIGHTING);

    glColor4f(1,0.6,0,0.2);

    glutWireSphere(5,40,40);
    glutWireSphere(6,40,40);

    glEnable(GL_LIGHTING);
}

void drawSaturnRing()
{
    glDisable(GL_LIGHTING);

    glColor3f(0.8,0.7,0.5);

    for(float r=1.8;r<2.7;r+=0.05)
    {
        glBegin(GL_LINE_LOOP);

        for(int i=0;i<360;i++)
        {
            float a=i*3.1416/180;

            glVertex3f(cos(a)*r,0,sin(a)*r);
        }

        glEnd();
    }

    glEnable(GL_LIGHTING);
}

void drawAsteroids()
{
    for(int i=0;i<ASTEROID_COUNT;i++)
    {
        glPushMatrix();

        glRotatef(asteroidAngle[i],0,1,0);
        glTranslatef(asteroidRadius[i],0,0);

        glColor3f(.5,.5,.5);
        glutSolidSphere(.15,8,8);

        glPopMatrix();
    }
}

void drawComet()
{
    glPushMatrix();

    glTranslatef(cometX,20,-50);

    glColor3f(1,1,1);
    glutSolidSphere(.5,10,10);

    glDisable(GL_LIGHTING);

    glBegin(GL_LINES);

    glColor3f(.5,.8,1);

    for(int i=0;i<20;i++)
        glVertex3f(-i*1.2,0,0);

    glEnd();

    glEnable(GL_LIGHTING);

    glPopMatrix();
}

void display()
{
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

    glLoadIdentity();

    float lx=sin(yaw);
    float lz=-cos(yaw);

    gluLookAt(camX,camY,camZ,
              camX+lx,camY,camZ+lz,
              0,1,0);

    drawStars();

    GLfloat lightpos[]={0,0,0,1};

    glLightfv(GL_LIGHT0,GL_POSITION,lightpos);

    drawSun();

    glPushMatrix();
    glRotatef(mercury,0,1,0);
    glTranslatef(8,0,0);
    glColor3f(.7,.7,.7);
    glutSolidSphere(.5,20,20);
    glPopMatrix();

    glPushMatrix();
    glRotatef(venus,0,1,0);
    glTranslatef(11,0,0);
    glColor3f(1,.5,0);
    glutSolidSphere(.7,20,20);
    glPopMatrix();

    glPushMatrix();

    glRotatef(earth,0,1,0);
    glTranslatef(15,0,0);

    glColor3f(.2,.5,1);
    glutSolidSphere(1,30,30);

    glPushMatrix();
    glRotatef(moon,0,1,0);
    glTranslatef(2,0,0);
    glColor3f(.8,.8,.8);
    glutSolidSphere(.3,20,20);
    glPopMatrix();

    glPushMatrix();
    glRotatef(satellite,0,1,0);
    glTranslatef(3,0,0);
    glutSolidCube(.4);
    glPopMatrix();

    glPopMatrix();

    glPushMatrix();
    glRotatef(mars,0,1,0);
    glTranslatef(20,0,0);
    glColor3f(1,0,0);
    glutSolidSphere(.7,20,20);
    glPopMatrix();

    drawAsteroids();

    glPushMatrix();
    glRotatef(saturn,0,1,0);
    glTranslatef(32,0,0);

    glColor3f(.9,.8,.5);
    glutSolidSphere(1.5,20,20);

    drawSaturnRing();

    glPopMatrix();

    drawComet();

    glutSwapBuffers();
}

void update(int v)
{
    mercury+=4;
    venus+=3;
    earth+=2;
    mars+=1.5;
    saturn+=.8;

    moon+=6;
    satellite+=4;

    cometX+=.7;

    if(cometX>200)
        cometX=-200;

    glutPostRedisplay();

    glutTimerFunc(16,update,0);
}

void keyboard(unsigned char key,int x,int y)
{
    float speed=2;

    if(key=='w')
    {
        camX+=sin(yaw)*speed;
        camZ-=cos(yaw)*speed;
    }

    if(key=='s')
    {
        camX-=sin(yaw)*speed;
        camZ+=cos(yaw)*speed;
    }

    if(key=='a')
        yaw-=.05;

    if(key=='d')
        yaw+=.05;
}

void init()
{
    glEnable(GL_DEPTH_TEST);

    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);

    glClearColor(0,0,0,1);

    initStars();
    initAsteroids();
}

void reshape(int w,int h)
{
    glViewport(0,0,w,h);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    gluPerspective(60,(float)w/h,1,2000);

    glMatrixMode(GL_MODELVIEW);
}

int main(int argc,char** argv)
{
    glutInit(&argc,argv);

    glutInitDisplayMode(GLUT_DOUBLE|GLUT_RGB|GLUT_DEPTH);

    glutInitWindowSize(1200,800);

    glutCreateWindow("Advanced Solar System Simulator");

    init();

    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutKeyboardFunc(keyboard);

    glutTimerFunc(16,update,0);

    glutMainLoop();

    return 0;
}
/*
🎮 Controls
Key	Action
W	Move forward
S	Move backward
A	Rotate left
D	Rotate right
*/

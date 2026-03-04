#include <GL/glut.h>
#include <math.h>
#include <stdlib.h>

float mercury=0, venus=0, earth=0, mars=0, jupiter=0, saturn=0;
float uranus=0, neptune=0, moon=0, satellite=0;

float earthRotation = 0;

float zoom = -70;

float camAngleX = 20;
float camAngleY = 30;

int lastX, lastY;
bool mouseDown=false;

GLuint earthTexture;

const int STAR_COUNT = 800;
float stars[STAR_COUNT][3];

void initStars()
{
    for(int i=0;i<STAR_COUNT;i++)
    {
        stars[i][0]=(rand()%400)-200;
        stars[i][1]=(rand()%400)-200;
        stars[i][2]=(rand()%400)-200;
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

void drawOrbit(float r)
{
    glDisable(GL_LIGHTING);

    glColor3f(.6,.6,.6);

    glBegin(GL_LINE_LOOP);
    for(int i=0;i<360;i++)
    {
        float a=i*3.1416/180;
        glVertex3f(cos(a)*r,0,sin(a)*r);
    }
    glEnd();

    glEnable(GL_LIGHTING);
}

void createEarthTexture()
{
    const int SIZE=64;
    GLubyte img[SIZE][SIZE][3];

    for(int i=0;i<SIZE;i++)
    for(int j=0;j<SIZE;j++)
    {
        if((i/8+j/8)%2==0)
        {
            img[i][j][0]=0;
            img[i][j][1]=120;
            img[i][j][2]=255;
        }
        else
        {
            img[i][j][0]=0;
            img[i][j][1]=180;
            img[i][j][2]=80;
        }
    }

    glGenTextures(1,&earthTexture);
    glBindTexture(GL_TEXTURE_2D,earthTexture);

    glTexImage2D(GL_TEXTURE_2D,0,GL_RGB,SIZE,SIZE,0,GL_RGB,GL_UNSIGNED_BYTE,img);

    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
}

void drawEarth()
{
    glEnable(GL_TEXTURE_2D);

    glBindTexture(GL_TEXTURE_2D,earthTexture);

    GLUquadric *q=gluNewQuadric();
    gluQuadricTexture(q,GL_TRUE);

    gluSphere(q,1,30,30);

    gluDeleteQuadric(q);

    glDisable(GL_TEXTURE_2D);
}

void drawSatellite()
{
    glColor3f(.8,.8,.8);

    glutSolidCube(.3);

    glPushMatrix();
    glTranslatef(.5,0,0);
    glScalef(1,.1,.6);
    glutSolidCube(.6);
    glPopMatrix();

    glPushMatrix();
    glTranslatef(-.5,0,0);
    glScalef(1,.1,.6);
    glutSolidCube(.6);
    glPopMatrix();
}

void drawSaturnRing()
{
    glDisable(GL_LIGHTING);

    glColor3f(.8,.7,.5);

    glBegin(GL_LINE_LOOP);
    for(int i=0;i<360;i++)
    {
        float a=i*3.1416/180;
        glVertex3f(cos(a)*2,0,sin(a)*2);
    }
    glEnd();

    glBegin(GL_LINE_LOOP);
    for(int i=0;i<360;i++)
    {
        float a=i*3.1416/180;
        glVertex3f(cos(a)*2.6,0,sin(a)*2.6);
    }
    glEnd();

    glEnable(GL_LIGHTING);
}

void display()
{
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

    glLoadIdentity();

    glTranslatef(0,0,zoom);

    glRotatef(camAngleX,1,0,0);
    glRotatef(camAngleY,0,1,0);

    drawStars();

    GLfloat light_pos[]={0,0,0,1};
    glLightfv(GL_LIGHT0,GL_POSITION,light_pos);

    // SUN
    glColor3f(1,1,.2);
    glutSolidSphere(3,50,50);

    // Mercury
    drawOrbit(8);
    glPushMatrix();
    glRotatef(mercury,0,1,0);
    glTranslatef(8,0,0);
    glColor3f(.6,.6,.6);
    glutSolidSphere(.5,20,20);
    glPopMatrix();

    // Venus
    drawOrbit(11);
    glPushMatrix();
    glRotatef(venus,0,1,0);
    glTranslatef(11,0,0);
    glColor3f(1,.5,0);
    glutSolidSphere(.7,20,20);
    glPopMatrix();

    // EARTH SYSTEM
    drawOrbit(15);
    glPushMatrix();

    glRotatef(earth,0,1,0);
    glTranslatef(15,0,0);

    glRotatef(earthRotation,0,1,0);

    drawEarth();

    // Moon
    glPushMatrix();
    glRotatef(moon,0,1,0);
    glTranslatef(2,0,0);
    glColor3f(.8,.8,.8);
    glutSolidSphere(.3,20,20);
    glPopMatrix();

    // Satellite
    glPushMatrix();
    glRotatef(satellite,0,1,0);
    glTranslatef(3,0,0);
    drawSatellite();
    glPopMatrix();

    glPopMatrix();

    // Mars
    drawOrbit(20);
    glPushMatrix();
    glRotatef(mars,0,1,0);
    glTranslatef(20,0,0);
    glColor3f(1,0,0);
    glutSolidSphere(.6,20,20);
    glPopMatrix();

    // Jupiter
    drawOrbit(26);
    glPushMatrix();
    glRotatef(jupiter,0,1,0);
    glTranslatef(26,0,0);
    glColor3f(.9,.6,.3);
    glutSolidSphere(1.5,20,20);
    glPopMatrix();

    // Saturn
    drawOrbit(33);
    glPushMatrix();
    glRotatef(saturn,0,1,0);
    glTranslatef(33,0,0);
    glColor3f(.9,.8,.5);
    glutSolidSphere(1.2,20,20);
    drawSaturnRing();
    glPopMatrix();

    glutSwapBuffers();
}

void update(int v)
{
    mercury+=4;
    venus+=3;
    earth+=2;
    mars+=1.6;
    jupiter+=1;
    saturn+=.8;

    moon+=6;
    satellite+=5;

    earthRotation+=2;

    glutPostRedisplay();

    glutTimerFunc(16,update,0);
}

void mouseMotion(int x,int y)
{
    if(mouseDown)
    {
        camAngleY+= (x-lastX)*0.5;
        camAngleX+= (y-lastY)*0.5;

        lastX=x;
        lastY=y;
    }
}

void mouseButton(int button,int state,int x,int y)
{
    if(button==GLUT_LEFT_BUTTON)
    {
        if(state==GLUT_DOWN)
        {
            mouseDown=true;
            lastX=x;
            lastY=y;
        }
        else
            mouseDown=false;
    }
}

void init()
{
    glEnable(GL_DEPTH_TEST);

    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);

    GLfloat ambient[]={.2,.2,.2,1};
    GLfloat diffuse[]={1,1,1,1};

    glLightfv(GL_LIGHT0,GL_AMBIENT,ambient);
    glLightfv(GL_LIGHT0,GL_DIFFUSE,diffuse);

    glClearColor(0,0,0,1);

    createEarthTexture();
    initStars();
}

void reshape(int w,int h)
{
    glViewport(0,0,w,h);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    gluPerspective(60,(float)w/h,1,1000);

    glMatrixMode(GL_MODELVIEW);
}

int main(int argc,char** argv)
{
    glutInit(&argc,argv);

    glutInitDisplayMode(GLUT_DOUBLE|GLUT_RGB|GLUT_DEPTH);

    glutInitWindowSize(1200,800);

    glutCreateWindow("Full Advanced Solar System - CG Project");

    init();

    glutDisplayFunc(display);
    glutReshapeFunc(reshape);

    glutMouseFunc(mouseButton);
    glutMotionFunc(mouseMotion);

    glutTimerFunc(16,update,0);

    glutMainLoop();

    return 0;
}

#include <GL/glut.h>
#include <math.h>
#include <stdlib.h>

float mercury=0, venus=0, earth=0, mars=0, jupiter=0, saturn=0;
float uranus=0, neptune=0, moon=0;

float speed = 0.2;
float zoom = -60;
float camAngle = 0;

const int STAR_COUNT = 200;
float stars[STAR_COUNT][3];

void initStars()
{
    for(int i=0;i<STAR_COUNT;i++)
    {
        stars[i][0]=(rand()%200)-100;
        stars[i][1]=(rand()%200)-100;
        stars[i][2]=(rand()%200)-100;
    }
}

void drawStars()
{
    glPointSize(2);
    glBegin(GL_POINTS);
    for(int i=0;i<STAR_COUNT;i++)
    {
        glColor3f(1,1,1);
        glVertex3f(stars[i][0],stars[i][1],stars[i][2]);
    }
    glEnd();
}

void drawOrbit(float r)
{
    glBegin(GL_LINE_LOOP);
    for(int i=0;i<360;i++)
    {
        float a=i*3.1416/180;
        glVertex3f(cos(a)*r,0,sin(a)*r);
    }
    glEnd();
}

void drawSaturnRing()
{
    glColor3f(0.8,0.7,0.5);
    glBegin(GL_LINE_LOOP);
    for(int i=0;i<360;i++)
    {
        float a=i*3.1416/180;
        glVertex3f(cos(a)*2.2,0,sin(a)*2.2);
    }
    glEnd();

    glBegin(GL_LINE_LOOP);
    for(int i=0;i<360;i++)
    {
        float a=i*3.1416/180;
        glVertex3f(cos(a)*2.8,0,sin(a)*2.8);
    }
    glEnd();
}

void display()
{
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();

    float camX = 80*sin(camAngle);
    float camZ = 80*cos(camAngle);

    gluLookAt(camX,30,camZ, 0,0,0, 0,1,0);

    drawStars();

    glPushMatrix();
    glTranslatef(0,0,zoom);

    // SUN
    glColor3f(1,0.8,0);
    glutSolidSphere(3,50,50);

    // Mercury
    drawOrbit(7);
    glPushMatrix();
    glRotatef(mercury,0,1,0);
    glTranslatef(7,0,0);
    glColor3f(0.7,0.7,0.7);
    glutSolidSphere(0.5,20,20);
    glPopMatrix();

    // Venus
    drawOrbit(10);
    glPushMatrix();
    glRotatef(venus,0,1,0);
    glTranslatef(10,0,0);
    glColor3f(1,0.5,0);
    glutSolidSphere(0.7,20,20);
    glPopMatrix();

    // Earth + Moon
    drawOrbit(14);
    glPushMatrix();
    glRotatef(earth,0,1,0);
    glTranslatef(14,0,0);

    glColor3f(0,0.4,1);
    glutSolidSphere(0.8,20,20);

    // Moon
    glRotatef(moon,0,1,0);
    glTranslatef(2,0,0);
    glColor3f(0.8,0.8,0.8);
    glutSolidSphere(0.3,20,20);

    glPopMatrix();

    // Mars
    drawOrbit(18);
    glPushMatrix();
    glRotatef(mars,0,1,0);
    glTranslatef(18,0,0);
    glColor3f(1,0,0);
    glutSolidSphere(0.6,20,20);
    glPopMatrix();

    // Jupiter
    drawOrbit(23);
    glPushMatrix();
    glRotatef(jupiter,0,1,0);
    glTranslatef(23,0,0);
    glColor3f(0.9,0.6,0.3);
    glutSolidSphere(1.5,20,20);
    glPopMatrix();

    // Saturn
    drawOrbit(29);
    glPushMatrix();
    glRotatef(saturn,0,1,0);
    glTranslatef(29,0,0);
    glColor3f(0.9,0.8,0.5);
    glutSolidSphere(1.2,20,20);
    drawSaturnRing();
    glPopMatrix();

    // Uranus
    drawOrbit(35);
    glPushMatrix();
    glRotatef(uranus,0,1,0);
    glTranslatef(35,0,0);
    glColor3f(0.5,0.8,1);
    glutSolidSphere(1,20,20);
    glPopMatrix();

    // Neptune
    drawOrbit(41);
    glPushMatrix();
    glRotatef(neptune,0,1,0);
    glTranslatef(41,0,0);
    glColor3f(0.2,0.3,1);
    glutSolidSphere(1,20,20);
    glPopMatrix();

    glPopMatrix();

    glutSwapBuffers();
}

void update(int v)
{
    mercury+=4*speed;
    venus+=3*speed;
    earth+=2.5*speed;
    mars+=2*speed;
    jupiter+=1.5*speed;
    saturn+=1.2*speed;
    uranus+=1*speed;
    neptune+=0.8*speed;

    moon+=6*speed;

    glutPostRedisplay();
    glutTimerFunc(16,update,0);
}

void keyboard(unsigned char key,int x,int y)
{
    switch(key)
    {
        case '+': zoom+=2; break;
        case '-': zoom-=2; break;

        case 'a': camAngle-=0.05; break;
        case 'd': camAngle+=0.05; break;

        case 'f': speed+=0.1; break;
        case 's': speed-=0.1; if(speed<0) speed=0; break;
    }
}

void reshape(int w,int h)
{
    glViewport(0,0,w,h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60,(float)w/h,1,1000);
    glMatrixMode(GL_MODELVIEW);
}

void init()
{
    glEnable(GL_DEPTH_TEST);
    glClearColor(0,0,0,1);
    initStars();
}

int main(int argc,char** argv)
{
    glutInit(&argc,argv);
    glutInitDisplayMode(GLUT_DOUBLE|GLUT_RGB|GLUT_DEPTH);
    glutInitWindowSize(1000,700);
    glutCreateWindow("Advanced Solar System Simulation");

    init();

    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutKeyboardFunc(keyboard);

    glutTimerFunc(16,update,0);

    glutMainLoop();
    return 0;
}

#include "framework.h"

const char* const vertexSource = R"(
	#version 330				// Shader 3.3
	precision highp float;		// normal floats, makes no difference on desktop computers

	uniform mat4 MVP;			// uniform variable, the Model-View-Projection transformation matrix
	layout(location = 0) in vec2 vp;
	layout(location = 1) in vec2 vertexUV;			// Attrib Array 1

	out vec2 texCoord;								// output attribute

	void main() {
		texCoord = vertexUV;			
		gl_Position = vec4(vp.x, vp.y, 0, 1) * MVP;		// transform vp from modeling space to normalized device space
	}
)";

const char* const fragmentSource = R"(
	#version 330			// Shader 3.3
	precision highp float;	// normal floats, makes no difference on desktop computers
	uniform sampler2D textureUnit;	

	in vec2 texCoord;
	out vec4 outColor;		// computed color of the current pixel

	void main() {
		outColor = texture(textureUnit, texCoord);	// computed color is the color of the primitive
		
	}
)";

GPUProgram gpuProgram; 
unsigned int vao[2];	   
unsigned int vbo[2];		
unsigned int vbl;			
float graph[100];			
int lineEnds[248];			
float linesCurrent[248];	
float korpontok[36200];
float circlevecs[722];
vec2 uvs[18100];
unsigned int textureId;
vec4 colors[50];
bool erovez = false;
static float tries =0 ;

float RandomFloat(float a, float b) {  ///forrás :https://stackoverflow.com/questions/5289613/generate-random-float-between-two-floats/5289624
	float random = ((float)rand()) / (float)RAND_MAX;
	float diff = b - a;
	float r = random * diff;
	return a + r;
}

void hiperbolikusbaTranszform() { 
	for (int i = 0; i < 100; i += 2) {
		float w = sqrtf(1 - powf(graph[i], 2) - powf(graph[i + 1], 2));
		graph[i] /= w;
		graph[i + 1] /= w;
	}
}

void euklideszbeTranszform() { 
	for (int i = 0; i < 100; i += 2) {
		float w = sqrtf(1 + powf(graph[i], 2) + powf(graph[i + 1], 2));
		graph[i] = graph[i] / w;
		graph[i + 1] = graph[i + 1] / w;
	}
}
void updateCircles() {  //https://www.youtube.com/watch?v=YDCSKlFqpaU forrás a kör létrehozáshoz.
	int points = 360;
	euklideszbeTranszform();



	for (int j = 0; j < 100; j += 2) {


		for (int i = 0; i <= points; i++) {
			float sugar = 0.0027;
			float radius = i * 2 * M_PI / 360;
			circlevecs[(i + 1) * 2] = sugar * cosf(radius) + graph[j];  
			circlevecs[(i + 1) * 2 + 1] = sugar * sinf(radius) + graph[j + 1];
			float circlew = sqrtf(1 - powf(circlevecs[(i + 1) * 2], 2) - powf(circlevecs[(i + 1) * 2 + 1], 2));
			circlevecs[(i + 1) * 2] /= circlew;  
			circlevecs[(i + 1) * 2 + 1] /= circlew;
		}
		float graphw = sqrtf(1 - powf(graph[j], 2) - powf(graph[j + 1], 2));
		graph[j] /= graphw;  
		graph[j + 1] /= graphw;
		graphw = sqrtf(1 + powf(graph[j], 2) + powf(graph[j + 1], 2));

		for (int i = 0; i <= points; i++) {  
			float circlew = sqrtf(1 + powf(circlevecs[(i + 1) * 2], 2) + powf(circlevecs[(i + 1) * 2 + 1], 2));
			float tav = acoshf(-(graph[j] * circlevecs[(i + 1) * 2] + graph[j + 1] * circlevecs[(i + 1) * 2 + 1] - graphw * circlew));
			circlevecs[(i + 1) * 2] = (circlevecs[(i + 1) * 2] - graph[j] * cosh(tav)) / sinh(tav);
			circlevecs[(i + 1) * 2 + 1] = (circlevecs[(i + 1) * 2 + 1] - graph[j + 1] * cosh(tav)) / sinh(tav);
		}
		int points = 360;

		korpontok[j * points] = graph[j];
		korpontok[j * points + 1] = graph[j + 1];
		float jw = sqrtf(1 + powf(graph[j], 2) + powf(graph[j + 1], 2));
		korpontok[j * points] /= graphw;
		korpontok[j * points + 1] /= graphw;

		for (int i = 0; i <= points; i++) { 
			float d = 0.035;
			float radius = i * 2 * M_PI / 360;
			korpontok[(i + 1) * 2 + j * 360] = graph[j] * cosh(d) + circlevecs[(i + 1) * 2] * sinh(d);
			korpontok[(i + 1) * 2 + 1 + j * 360] = graph[j + 1] * cosh(d) + circlevecs[(i + 1) * 2 + 1] * sinh(d);
			float w = sqrtf(1 + powf(korpontok[(i + 1) * 2 + j * 360], 2) + powf(korpontok[(i + 1) * 2 + 1 + j * 360], 2));
			korpontok[(i + 1) * 2 + j * 360] /= w;
			korpontok[(i + 1) * 2 + 1 + j * 360] /= w;
		}
		graph[j] /= graphw;  
		graph[j + 1] /= graphw;

	}
	hiperbolikusbaTranszform();

}

void updateLines() {
	for (int i = 0; i < 248; i += 2) {
		int x = lineEnds[i];
		linesCurrent[i] = graph[x];
		linesCurrent[i + 1] = graph[x + 1];
	}
}

void heurisztika() {
	float tempArray[100];
	int nevezo[50];
	for (int i = 0; i < 100; i += 2) {
		tempArray[i] = 0;
		tempArray[i + 1] = 0;
		nevezo[i / 2] = 0;
	}

	bool vanel = false;
	hiperbolikusbaTranszform();
	for (int i = 0; i < 100; i += 2) {
		float xAtlag = 0;
		float yAtlag = 0;
		for (int j = 0; j < 100; j += 2) {
			for (int x = 0; x < 248; x += 4) {
				if (i != j) {
					if (i == lineEnds[x] && j == lineEnds[x + 2]) {
						xAtlag += graph[j];
						yAtlag += graph[j + 1];
						nevezo[i / 2] += 1;
						vanel = true;
					}
					else if (i == lineEnds[x + 2] && j == lineEnds[x]) {
						xAtlag += graph[j];
						yAtlag += graph[j + 1];
						vanel = true;
						nevezo[i / 2] += 1;
					}
				}

			}
			if (vanel == false && i != j) {
				xAtlag -= graph[j];
				yAtlag -= graph[j + 1];
				nevezo[i / 2] -= 1;
			}
			vanel = false;


		}
		tempArray[i] = xAtlag;
		tempArray[i + 1] = yAtlag;

	}
	for (int i = 0; i < 100; i += 2) {
		graph[i] -= tempArray[i] / nevezo[i / 2];
		graph[i + 1] -= tempArray[i + 1] / nevezo[i / 2];
	}

	updateCircles();
	euklideszbeTranszform();
	updateLines();

}

void erovezerelt(float time) {

	float tempArray[100];   

	for (int i = 0; i < 100; i += 2) {
		tempArray[i] = 0;
		tempArray[i + 1] = 0;
	}
	hiperbolikusbaTranszform();

	bool vanel = false;
	for (int i = 0; i < 100; i += 2) {
		for (int j = 0; j < 100; j += 2) {
			float xAtlag = 0;
			float yAtlag = 0;
			float tav = 0;
			float d = 0.4;
			float v1 = 0;
			float v2 = 0;
			float v3 = 0;
			float wi = sqrtf(1 + powf(graph[i], 2) + powf(graph[i + 1], 2));
			float wj = sqrtf(1 + powf(graph[j], 2) + powf(graph[j + 1], 2));
			for (int x = 0; x < 248; x += 4) {
				if (i != j) {
					if (i == lineEnds[x] && j == lineEnds[x + 2]) {
						tav = acoshf(-(graph[i] * graph[j] + graph[i + 1] * graph[j + 1] - wi * wj));
						v1 = (graph[j] - graph[i] * cosh(tav)) / sinh(tav);
						v2 = (graph[j + 1] - graph[i + 1] * cosh(tav)) / sinh(tav);
						v3 = (wj - wi * cosh(tav)) / sinh(tav);
						vanel = true;
						break;

					}
					else if (i == lineEnds[x + 2] && j == lineEnds[x]) {
						tav = acoshf(-(graph[i] * graph[j] + graph[i + 1] * graph[j + 1] - sqrtf(1 + powf(graph[i], 2) + powf(graph[i + 1], 2)) * sqrtf(1 + powf(graph[j], 2) + powf(graph[j + 1], 2))));
						v1 = (graph[j] - graph[i] * cosh(tav)) / sinh(tav);
						v2 = (graph[j + 1] - graph[i + 1] * cosh(tav)) / sinh(tav);
						vanel = true;
						break;
					}
				}

			}
			if (vanel == true && i != j) { 
				if (tav > d) {
					tempArray[i] += powf(v1 - d, 3);
					tempArray[i + 1] += powf(v2 - d, 3);
				}
				else {
					tempArray[i] -= powf(v1 - d, 3);
					tempArray[i + 1] -= powf(v2 - d, 3);
				}

			}
			else if (vanel == false && i != j) {
				tav = acoshf(-(graph[i] * graph[j] + graph[i + 1] * graph[j + 1] - sqrtf(1 + powf(graph[i], 2) + powf(graph[i + 1], 2)) * sqrtf(1 + powf(graph[j], 2) + powf(graph[j + 1], 2))));
				if (tav < 0.4) {
					v1 = (graph[j] - graph[i] * cosh(tav)) / sinh(tav);
					v2 = (graph[j + 1] - graph[i + 1] * cosh(tav)) / sinh(tav);
					tempArray[i] -= v1 / (tav );
					tempArray[i + 1] -= v2 / (tav );
				}
			}
			vanel = false;


		}

	}

	for (int i = 0; i < 100; i += 2) {
		graph[i] = graph[i] * cosh(0.01*(20 -time)/ 20) + tempArray[i] * sinh(0.01*(20 - time) / 20);
		graph[i + 1] = graph[i + 1] * cosh(0.01 * (20 - time) / 20) + tempArray[i + 1] * sinh(0.01 * (20 - time) / 20);
	}

	for (int i = 0; i < 100; i += 2) {		
		float tav = acoshf(-(graph[i] * 0 + graph[i + 1] * 0 - sqrtf(1 + powf(graph[i], 2) + powf(graph[i + 1], 2)) * 1));
		float vx = (0 - graph[i] * cosh(tav)) / sinh(tav);
		float vy = (0 - graph[i+1] * cosh(tav)) / sinh(tav);
		graph[i] = graph[i] * cosh(0.01 * tav) + vx * sinh(0.01 * tav);
		graph[i+1] = graph[i+1] * cosh(0.01 * tav) + vy * sinh(0.01 * tav);
	}

	updateCircles();
	euklideszbeTranszform();
	updateLines();

	erovez = true;
}

void UploadTexture(int width, int height, std::vector<vec4>& image) {  

	glGenTextures(1, &textureId);
	glBindTexture(GL_TEXTURE_2D, textureId);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_FLOAT, &image[0]);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
}


void initColors() {
	for (int j = 0; j < 50; j++) {
		colors[j]=vec4(RandomFloat(0, 1), RandomFloat(0, 1), RandomFloat(0, 1), 1);
	}
}



void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	initColors();

	for (int i = 0; i < 100; i++) {  
		graph[i] = RandomFloat(-1, 1);
	}
	updateCircles();
	euklideszbeTranszform();


	for (int i = 0; i < 248; i += 4) {   
		int x = (rand() % (50)) * 2;
		int y = (rand() % (50)) * 2;
		lineEnds[i] = x;
		lineEnds[i + 1] = x + 1;
		lineEnds[i + 2] = y;
		lineEnds[i + 3] = y + 1;
		for (int j = 0; j < i; j++) {
			if ((x == lineEnds[j] && x + 1 == lineEnds[j + 1] && y == lineEnds[j + 2] && y + 1 == lineEnds[j + 3]) ||
				(x + 1 == lineEnds[j + 3] && x == lineEnds[j + 2] && y == lineEnds[j] && x + 1 == lineEnds[j + 1])
				) {
				i -= 2;
				break;
			}
		}
	}

	updateLines();

	glGenVertexArrays(2, vao);
	glBindVertexArray(vao[0]);

	glGenBuffers(2, vbo);		
	glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(korpontok), korpontok, GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);

	glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(uvs), uvs, GL_STATIC_DRAW);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, NULL);

	glBindVertexArray(vao[1]);		
	glGenBuffers(1, &vbl);
	glBindBuffer(GL_ARRAY_BUFFER, vbl);
	glBufferData(GL_ARRAY_BUFFER, sizeof(linesCurrent), linesCurrent, GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	
	gpuProgram.create(vertexSource, fragmentSource, "outColor");
}


void onDisplay() {
	glClearColor(0, 0, 0, 0);     
	glClear(GL_COLOR_BUFFER_BIT); 


	float MVPtransf[4][4] = { 1, 0, 0, 0,    
							  0, 1, 0, 0,    
							  0, 0, 1, 0,
							  0, 0, 0, 1 };

	int location = glGetUniformLocation(gpuProgram.getId(), "MVP");	
	glUniformMatrix4fv(location, 1, GL_TRUE, &MVPtransf[0][0]);	

	glBindVertexArray(vao[0]);
	glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(korpontok), korpontok, GL_STATIC_DRAW);


	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0,       
		2, GL_FLOAT, GL_FALSE, 
		0, NULL); 		     

	for (int j = 0; j < 50; j++) {
		int width = 10, height = 10;					
		std::vector<vec4> image(width * height);
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				image[y * width + x] =colors[j];
			}
		}
		UploadTexture(width, height, image);				
		int sampler = 0;
		int location = glGetUniformLocation(gpuProgram.getId(), "textureUnit");
		glUniform1i(location, sampler);

		glActiveTexture(GL_TEXTURE0 + sampler);
		glBindTexture(GL_TEXTURE_2D, textureId);
		glDrawArrays(GL_TRIANGLE_FAN, j * 360, 360);
	}

	glBindVertexArray(vao[1]);
	glBindBuffer(GL_ARRAY_BUFFER, vbl);
	glBufferData(GL_ARRAY_BUFFER, sizeof(linesCurrent), linesCurrent, GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);

	int width = 10, height = 10;
	std::vector<vec4> image(width * height);
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			image[y * width + x] = vec4(1,1,0,1);
		}
	}
	UploadTexture(width, height, image);				
	int sampler = 0;
	location = glGetUniformLocation(gpuProgram.getId(), "textureUnit");
	glUniform1i(location, sampler);

	glActiveTexture(GL_TEXTURE0 + sampler);
	glBindTexture(GL_TEXTURE_2D, textureId);
	glDrawArrays(GL_LINES, 0, 124);

	glBindVertexArray(0);

	glutSwapBuffers(); 
}


void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == ' ') {
		for (int i = 0; i < 5; i++) {
			heurisztika();
		};  glutPostRedisplay();  erovez = true;tries = 0;   
	}
	
}


void onKeyboardUp(unsigned char key, int pX, int pY) {

}
float cX = 0;   
float cY = 0;

void onMouseMotion(int pX, int pY) {	

	float aftercX = (2.0f * pX / windowWidth - 1) - cX;	
	float aftercY = (1.0f - 2.0f * pY / windowHeight) - cY;
	float aftercW = sqrtf(1 - powf(aftercX, 2) - powf(aftercY, 2));
	aftercX /= aftercW;
	aftercY /= aftercW;
	aftercW = sqrtf(1 + powf(aftercX, 2) + powf(aftercY, 2));
	float dOP = acoshf(-(0 * aftercX + 0 * aftercY - 1 * aftercW)); 
	if (dOP > 0) {

		float tempgraph[100];
		for (int i = 0; i < 100; i++) {
			tempgraph[i] = 0;
		}
		float vx = (aftercX - 0 * cosh(dOP)) / sinh(dOP);
		float vy = (aftercY - 0 * cosh(dOP)) / sinh(dOP);
		float vw = (aftercW - 1 * cosh(dOP)) / sinh(dOP);
		vx /= sqrtf(vx * vx + vy * vy + vw * vw); 
		vy /= sqrtf(vx * vx + vy * vy + vw * vw);
		vw /= sqrtf(vx * vx + vy * vy + vw * vw);

		float m1x = 0;	
		float m1y = 0;
		float m1w = 1;

		float m2x = (0 * cosh(dOP / 2)) + vx * sinh(dOP / 2);	
		float m2y = (0 * cosh(dOP / 2)) + vy * sinh(dOP / 2);
		float m2w = (1 * cosh(dOP / 2)) + vw * sinh(dOP / 2);

		hiperbolikusbaTranszform();

		for (int i = 0; i < 100; i += 2) { 
			float pW = sqrtf(1 + powf(graph[i], 2) + powf(graph[i + 1], 2));
			float d1 = acoshf(-(graph[i] * m1x + graph[i + 1] * m1y - pW * m1w));
			float v1x = (m1x - graph[i] * cosh(d1)) / sinh(d1);
			float v1y = (m1y - graph[i + 1] * cosh(d1)) / sinh(d1);
			float v1w = (m1w - pW * cosh(d1)) / sinh(d1);
			tempgraph[i] = graph[i] * cosh(d1 * 2) + v1x * sinh(d1 * 2);
			tempgraph[i + 1] = graph[i + 1] * cosh(d1 * 2) + v1y * sinh(d1 * 2);
		}

		for (int i = 0; i < 100; i += 2) {  
			float pW = sqrtf(1 + powf(tempgraph[i], 2) + powf(tempgraph[i + 1], 2));
			float d1 = acoshf(-(tempgraph[i] * m2x + tempgraph[i + 1] * m2y - pW * m2w));
			float v1x = (m2x - tempgraph[i] * cosh(d1)) / sinh(d1);
			float v1y = (m2y - tempgraph[i + 1] * cosh(d1)) / sinh(d1);
			float v1w = (m2w - pW * cosh(d1)) / sinh(d1);
			tempgraph[i] = tempgraph[i] * cosh(d1 * 2) + v1x * sinh(d1 * 2);
			tempgraph[i + 1] = tempgraph[i + 1] * cosh(d1 * 2) + v1y * sinh(d1 * 2);
		}

		bool valid = true;
		for (int i = 0; i < 100; i++) {
			if (isnan(tempgraph[i])) {
				valid = false;
			}
		}
		if (valid == true) {
			for (int i = 0; i < 100; i++) {
				graph[i] = tempgraph[i];
			}
		}

		updateCircles();
		euklideszbeTranszform();

		updateLines();
	}
	cX = 2.0f * pX / windowWidth - 1;	
	cY = 1.0f - 2.0f * pY / windowHeight;
	onDisplay();
}


void onMouse(int button, int state, int pX, int pY) { 
	cX = 2.0f * pX / windowWidth - 1;	
	cY = 1.0f - 2.0f * pY / windowHeight;
	onDisplay();
}


void onIdle() {
	float time = glutGet(GLUT_ELAPSED_TIME); 
	if (erovez == true) {
		if (tries < 20) {
			erovezerelt(tries);
			glutPostRedisplay();
			tries += 1;
		}
	}
}


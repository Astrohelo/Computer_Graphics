#include "framework.h"
 
const char* vertexSource = R"(
	#version 330
    precision highp float;
 
	uniform vec3 wLookAt, wRight, wUp;          
 
	layout(location = 0) in vec2 cCamWindowVertex;	
	out vec3 p;
 
	void main() {
		gl_Position = vec4(cCamWindowVertex, 0, 1);
		p = wLookAt + wRight * cCamWindowVertex.x + wUp * cCamWindowVertex.y;
	}
)";
const char* fragmentSource = R"(
	#version 330
    precision highp float;
 
	struct Material {
		vec3 ka, kd, ks;
		float  shininess;
		vec3 F0;
		int rough, reflective;
	};
 
	struct Light {
		vec3 direction;
		vec3 Le, La;
	};
 
	struct FalCsucs{
		vec3 csucs;
	};
 
	struct FalCsucsIndex{
		int csucsindex;
	};
	
 
	struct Hit {
		float t;
		vec3 position, normal;
		int mat;	
	};
 
	struct Ray {
		vec3 start, dir;
	};
	
	const float epsilon = 0.0001f;
	
	uniform FalCsucs v[20];
	uniform FalCsucsIndex planes[60];
	
	uniform vec3 wEye; 
	uniform Light light;     
	uniform Material materials[3];  
	bool dodeka;
 
	in  vec3 p;					
	out vec4 fragmentColor;		
	
	vec4 quaternion(vec3 axis, float angle){
		vec4 q;
		float half_ang = (angle / 2) * 3.14159265 / 180.0;
		q.x = axis.x * sin(half_ang);
		q.y = axis.y * sin(half_ang);
		q.z = axis.z * sin(half_ang);
		q.w = cos(half_ang);
		return q;
	}
 
	vec4 qmul(vec4 q1, vec4 q2) {	
		return vec4(q1.w * q2.xyz + q2.w * q1.xyz + cross(q1.xyz, q2.xyz), q1.w * q2.w - dot(q1.xyz, q2.xyz));
	} 
 
	vec3 rotate(vec3 u, vec4 q){
		vec4 q_inv = vec4(-q.x,-q.y, -q.z, q.w);
		q = qmul(qmul(q, vec4(u.x, u.y, u.z, 0)), q_inv);
		return vec3(q.x,q.y,q.z);
	}
 
	void getPlane(int i, out vec3 p, out vec3 normal){
		vec3 p1= v[planes[5*i].csucsindex-1].csucs, p2= v[planes[5*i+1].csucsindex-1].csucs, p3 = v[planes[5*i+2].csucsindex-1].csucs;
		normal = cross(p2-p1,p3-p1);
		if(dot(p1, normal) < 0) normal = -normal; 
		p = p1 + vec3(0, 0,0.03f);
	}
 
	bool dodekaSide(Hit hit){		
		for(int i = 0; i<12;i++){
			for(int j = 0; j<5;j++){
				vec3 p1,p2;
				if(j!=4){
					p1= v[planes[5*i+j].csucsindex-1].csucs, p2= v[planes[5*i+j+1].csucsindex-1].csucs;
				}
				else if(j==4){
					p1= v[planes[5*i].csucsindex-1].csucs, p2= v[planes[5*i+j].csucsindex-1].csucs;
				}
				p1=vec3(p1.x, p1.y, p1.z);
				p2=vec3(p2.x, p2.y, p2.z);
				vec3 egyik = hit.position-p1;
				vec3 masik = p2-p1;
 
				float veglet = length(cross(egyik,masik));
 
				if (veglet<0.1){
					return true;
				}
			}
		}
		return false;
	}
 
	Hit intersectDodeka(Ray ray,Hit hit, int mat){
		
		for(int i = 0; i<12;i++){
			vec3 p1,normal;
			getPlane(i,p1,normal);
			float ti = abs(dot(normal,ray.dir))>epsilon? dot(p1-ray.start,normal)/dot(normal,ray.dir) : -1;
			if(ti <= epsilon || (ti>hit.t && hit.t>0)) continue;
			vec3 pintersect = ray.start + ray.dir * ti;
			bool outside = false;
			for(int j = 0; j<12; j++){
				if(i==j) continue;
				vec3 p11,n;
				getPlane(j,p11,n);
				if(dot(n,pintersect - p11)>0){
					outside=true;
					break;
				}
			}
			if(!outside){
				hit.t=ti;
				hit.position=pintersect;
				hit.normal = normalize(normal);
				hit.mat = mat;
				dodeka=true;
			}
		}
		return hit;
	}
 
	Hit intersect(Ray ray) {
		Hit hit;
		hit.t = -1;
 
		float a=2.3, b=1.3, c=1.3;
		vec3 str = ray.start;
		vec3 dir= normalize(ray.dir);
		float A = a*dir.x*dir.x + b*dir.y*dir.y;
		float B = 2*a*str.x*dir.x + 2*b*str.y*dir.y - c* dir.z;
		float C = a*str.x*str.x + b* str.y*str.y - c* str.z;
		float discr = B * B - 4.0f * A * C;
 
		if (discr < 0) return hit;
		float sqrt_discr = sqrt(discr);
		float t1 = (-B + sqrt_discr) / 2.0f / A;	
		float t2 = (-B - sqrt_discr) / 2.0f / A;
		if (t1 <= 0 && t2 <= 0) return hit;
		
		vec3 temp1 = ray.start + ray.dir * t1;
		vec3 temp2 = ray.start + ray.dir * t2;
 
		float gombtav1 = distance(temp1,vec3(0,0,0));
		float gombtav2 = distance(temp2,vec3(0,0,0));
 
		if (gombtav1>0.3f && gombtav2> 0.3f){ 
			hit.t= -1;
			return hit;
		}
		else if(gombtav1>0.3f && gombtav2 < 0.3f){
			hit.t=t2;
		}
		else if(gombtav1<0.3f && gombtav2 > 0.3f){
			hit.t=t1;
		}
		else{
			hit.t = (t2 > 0) ? t2 : t1;
			if(t2>t1){
				hit.t=t1;
			}
		}
		hit.position = ray.start + ray.dir * hit.t;
 
		vec3 X  = vec3(1, 0, 2*a*hit.position.x/c);
		vec3 egyik = X-hit.position ;
		vec3 Y  = vec3(0, 1, 2* b*hit.position.y/c);
		vec3 masik = Y-hit.position ;
		
		hit.normal = normalize(cross(egyik,masik));
		return hit;
	}
 
	Hit firstIntersect(Ray ray) {
		dodeka=false;
		Hit bestHit;
		bestHit.t = -1;
		
		Hit hit = intersect(ray); 
		hit.mat=1;
		if (hit.t > 0 )  bestHit = hit;
 
		bestHit = intersectDodeka(ray,bestHit,2);
		if(dodekaSide(bestHit)==true){bestHit.mat=0;};
		
		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}
 
	bool shadowIntersect(Ray ray) {	
		if (intersect( ray).t > 0) return true; 
		return false;
	}
 
	vec3 Fresnel(vec3 F0, float cosTheta) { 
		return F0 + (vec3(1, 1, 1) - F0) * pow(cosTheta, 5);
	}
 
 
 
	vec3 trace(Ray ray) {
		vec3 weight = vec3(1, 1, 1);
		vec3 outRadiance = vec3(0, 0, 0);
		int maxdepth= 5;
		for(int d = 0; d < maxdepth; d++) {
			Hit hit = firstIntersect(ray);
			if (hit.t < 0) return weight * light.La;
			if (materials[hit.mat].rough == 1) {
				outRadiance += weight * materials[hit.mat].ka * light.La;
				Ray shadowRay;
				shadowRay.start = hit.position + hit.normal * epsilon;
				shadowRay.dir = light.direction;
				float cosTheta = dot(hit.normal, light.direction);
				if (cosTheta > 0 && !shadowIntersect(shadowRay)) {
					outRadiance += weight * light.Le * materials[hit.mat].kd * cosTheta;
					vec3 halfway = normalize(-ray.dir + light.direction);
					float cosDelta = dot(hit.normal, halfway);
					if (cosDelta > 0) outRadiance += weight * light.Le * materials[hit.mat].ks * pow(cosDelta, materials[hit.mat].shininess);
				}
			}
 
			if (materials[hit.mat].reflective == 1) {
				weight *= Fresnel(materials[hit.mat].F0, dot(-ray.dir, hit.normal)); 
				if(dodeka==true){
					ray.start = rotate(hit.position + hit.normal * epsilon, quaternion(hit.normal,72)); 
					ray.dir = rotate(reflect(ray.dir, hit.normal),quaternion(hit.normal,72));
				}
				else if(dodeka==false){
					maxdepth++;
					ray.start = hit.position + hit.normal * epsilon;
					ray.dir = reflect(ray.dir, hit.normal);
				}
			} else return outRadiance;
		}
		
		return outRadiance+weight*light.La;
		
	}
 
	void main() {
		Ray ray;
		ray.start = wEye; 
		ray.dir = normalize(p - wEye);
		fragmentColor = vec4(trace(ray), 1); 
	}
)";
 
 
 
struct Material {
	
	vec3 ka, kd, ks;
	float  shininess;
	vec3 F0;
	int rough, reflective;
};
 
 
struct RoughMaterial : Material {
	RoughMaterial(vec3 _kd, vec3 _ks, float _shininess) {
		ka = _kd * M_PI;
		kd = _kd;
		ks = _ks;
		shininess = _shininess;
		rough = true;
		reflective = false;
	}
};
 
 
struct SmoothMaterial : Material {
	SmoothMaterial(vec3 _F0) {
		F0 = _F0;
		rough = false;
		reflective = true;
	}
};
 
 
 
struct FalCsucs {
	vec3 csucs;
	FalCsucs(const vec3& _csucs ) { csucs = _csucs; }
};
 
struct FalCsucsIndex {
	int csucsindex;
	FalCsucsIndex(float egy ) { 
		csucsindex = egy; 
	}
};
 
 
struct Camera {
	vec3 eye, lookat, right, up;
	float fov;
public:
	void set(vec3 _eye, vec3 _lookat, vec3 vup, float _fov) {
		eye = _eye;
		lookat = _lookat;
		fov = _fov;
		vec3 w = eye - lookat;
		float f = length(w);
		right = normalize(cross(vup, w)) * f * tanf(fov / 2);
		up = normalize(cross(w, right)) * f * tanf(fov / 2);
	}
	void Animate(float dt) {
		eye = vec3((eye.x - lookat.x) * cos(dt) + (eye.z - lookat.z) * sin(dt) + lookat.x,
			eye.y,
			-(eye.x - lookat.x) * sin(dt) + (eye.z - lookat.z) * cos(dt) + lookat.z);
		set(eye, lookat, up, fov);
	}
};
 
struct Light {
	vec3 direction;
	vec3 Le, La;
	Light(vec3 _direction, vec3 _Le, vec3 _La) {
		direction = normalize(_direction);
		Le = _Le; La = _La;
	}
};
 
class Shader : public GPUProgram {
public:
	void setUniformMaterials(const std::vector<Material*>& materials) {
		char name[256];
		for (unsigned int mat = 0; mat < materials.size(); mat++) {
			sprintf(name, "materials[%d].ka", mat); setUniform(materials[mat]->ka, name);
			sprintf(name, "materials[%d].kd", mat); setUniform(materials[mat]->kd, name);
			sprintf(name, "materials[%d].ks", mat); setUniform(materials[mat]->ks, name);
			sprintf(name, "materials[%d].shininess", mat); setUniform(materials[mat]->shininess, name);
			sprintf(name, "materials[%d].F0", mat); setUniform(materials[mat]->F0, name);
			sprintf(name, "materials[%d].rough", mat); setUniform(materials[mat]->rough, name);
			sprintf(name, "materials[%d].reflective", mat); setUniform(materials[mat]->reflective, name);
		}
	}
 
	void setUniformLight(Light* light) {
		setUniform(light->La, "light.La");
		setUniform(light->Le, "light.Le");
		setUniform(light->direction, "light.direction");
	}
 
	void setUniformCamera(const Camera& camera) {
		setUniform(camera.eye, "wEye");
		setUniform(camera.lookat, "wLookAt");
		setUniform(camera.right, "wRight");
		setUniform(camera.up, "wUp");
	}
 
 
	void setUniformFal(const std::vector<FalCsucs*>& v, const std::vector<FalCsucsIndex*>& planes) {
		char name[256];
		char nameplan[256];
		for (unsigned int o = 0; o < v.size(); o++) {
			sprintf(name, "v[%d].csucs", o); setUniform(v[o]->csucs, name);
		}
		for (unsigned int o = 0; o < planes.size(); o++) {
			sprintf(nameplan, "planes[%d].csucsindex", o); setUniform(planes[o]->csucsindex, nameplan);
		}
	}
};
 
 
float F(float n, float k) { return((n - 1) * (n - 1) + k * k) / ((n + 1) * (n + 1) + k * k); }
 
 
class Scene {
	std::vector<Light*> lights;
	Camera camera;
	std::vector<Material*> materials;
	std::vector<FalCsucs*> v;
	std::vector<FalCsucsIndex*> planes;
public:
	void build() {
		vec3 eye = vec3(0, 0, 1.3);
		vec3 vup = vec3(0, 1, 0);
		vec3 lookat = vec3(0, 0, 0);
		float fov = 50* (float)M_PI / 180;
		camera.set(eye, lookat, vup, fov);
 
		lights.push_back(new Light(vec3(1, 1, 1), vec3(3, 2, 2), vec3(0.4f, 0.3f, 0.4f)));
 
		vec3 kd(0.2f, 0.3f, 0.2f), ks(10, 10, 10);
		materials.push_back(new RoughMaterial(kd, ks, 500));
		materials.push_back(new SmoothMaterial(vec3(F(0.17f , 3.1f),F( 0.35f , 2.7f),F( 1.5f , 1.9f))));
		materials.push_back(new SmoothMaterial(vec3(1,1,1)));
		const float g = 0.618f, G = 1.618f;
		vec3 tempv[20] = { vec3(0,g,G),
		vec3(0,-g,G),
		vec3(0,-g,-G),
		vec3(0,g,-G),
		vec3(G,0,g),
		vec3(-G,0,g),
		vec3(-G,0,-g),
		vec3(G,0,-g),
		vec3(g,G,0),
		vec3(-g,G,0),
		vec3(-g,-G,0),
		vec3(g,-G,0),
		vec3(1,1,1),
		vec3(-1,1,1),
		vec3(-1,-1,1),
		vec3(1,-1,1),
		vec3(1,-1,-1),
		vec3(1,1,-1),
		vec3(-1,1,-1),
		vec3(-1,-1,-1) };
		for (int i = 0; i < 20; i++) {
			v.push_back(new FalCsucs(tempv[i]));
		};
 
 
		int tempplanes[60] = {
		1,2,16,5,13,
		1,13,9,10,14,
		1,14,6,15,2,
		2,15,11,12,16,
		3,4,18,8,17,
		3,17,12,11,20,
		3,20,7,19,4,
		19,10,9,18,4,
		16,12,17,8,5,
		5,8,18,9,13,
		14,10,19,7,6,
		6,7,20,11,15
		};
		for (int i = 0; i < 60; i++) {
			planes.push_back(new FalCsucsIndex(tempplanes[i]));
		};
 
	}
 
	void setUniform(Shader& shader) {
		shader.setUniformMaterials(materials);
		shader.setUniformLight(lights[0]);
		shader.setUniformCamera(camera);
		shader.setUniformFal(v, planes);
	}
 
	void Animate(float dt) { camera.Animate(dt); }
};
 
Shader shader; 
Scene scene;
 
 
class FullScreenTexturedQuad {
	unsigned int vao = 0;	
public:
	void create() {
		glGenVertexArrays(1, &vao);	
		glBindVertexArray(vao);		
 
		unsigned int vbo;		
		glGenBuffers(1, &vbo);	
 
		glBindBuffer(GL_ARRAY_BUFFER, vbo); 
		float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };	
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);	  
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);    
	}
 
	void Draw() {
		glBindVertexArray(vao);	
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);	
	}
};
 
FullScreenTexturedQuad fullScreenTexturedQuad;
 
 
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	scene.build();
	fullScreenTexturedQuad.create();
 
	shader.create(vertexSource, fragmentSource, "fragmentColor");
	shader.Use();
	
}
 
 
void onDisplay() {
	
 
	glClearColor(1.0f, 0.5f, 0.8f, 1.0f);							
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); 
 
	scene.setUniform(shader);
	fullScreenTexturedQuad.Draw();
 
	glutSwapBuffers();
}
 
 
void onKeyboard(unsigned char key, int pX, int pY) {
}
 
 
void onKeyboardUp(unsigned char key, int pX, int pY) {
 
}
 
 
void onMouse(int button, int state, int pX, int pY) {
}
 
 
void onMouseMotion(int pX, int pY) {
}
 
void onIdle() {
	scene.Animate(0.007f);
	glutPostRedisplay();
}
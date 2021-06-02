#include "framework.h"

template<class T> struct Dnum { 
	float f; 
	T d;  
	Dnum(float f0 = 0, T d0 = T(0)) { f = f0, d = d0; }
	Dnum operator+(Dnum r) { return Dnum(f + r.f, d + r.d); }
	Dnum operator-(Dnum r) { return Dnum(f - r.f, d - r.d); }
	Dnum operator*(Dnum r) {
		return Dnum(f * r.f, f * r.d + d * r.f);
	}
	Dnum operator/(Dnum r) {
		return Dnum(f / r.f, (r.f * d - r.d * f) / r.f / r.f);
	}
};


template<class T> Dnum<T> Exp(Dnum<T> g) { return Dnum<T>(expf(g.f), expf(g.f) * g.d); }
template<class T> Dnum<T> Sin(Dnum<T> g) { return  Dnum<T>(sinf(g.f), cosf(g.f) * g.d); }
template<class T> Dnum<T> Cos(Dnum<T>  g) { return  Dnum<T>(cosf(g.f), -sinf(g.f) * g.d); }
template<class T> Dnum<T> Tan(Dnum<T>  g) { return Sin(g) / Cos(g); }
template<class T> Dnum<T> Sinh(Dnum<T> g) { return  Dnum<T>(sinh(g.f), cosh(g.f) * g.d); }
template<class T> Dnum<T> Cosh(Dnum<T> g) { return  Dnum<T>(cosh(g.f), sinh(g.f) * g.d); }
template<class T> Dnum<T> Tanh(Dnum<T> g) { return Sinh(g) / Cosh(g); }
template<class T> Dnum<T> Log(Dnum<T> g) { return  Dnum<T>(logf(g.f), g.d / g.f); }
template<class T> Dnum<T> Pow(Dnum<T> g, float n) {
	return  Dnum<T>(powf(g.f, n), n * powf(g.f, n - 1) * g.d);
}

typedef Dnum<vec2> Dnum2;

const int tessellationLevel = 100;


struct Camera { 
	vec3 wEye, wLookat, wVup;   
	float fov, asp, fp, bp;		
public:
	Camera() {
		asp = (float)windowWidth / windowHeight;
		fov = 60 * (float)M_PI / 180.0f;
		fp = 0.1; bp = 20;
	}
	mat4 V() { 
		vec3 w = normalize(wEye - wLookat);
		vec3 u = normalize(cross(wVup, w));
		vec3 v = cross(w, u);
		return TranslateMatrix(wEye * (-1)) * mat4(u.x, v.x, w.x, 0,
			u.y, v.y, w.y, 0,
			u.z, v.z, w.z, 0,
			0, 0, 0, 1);
	}

	mat4 P() { 
		return mat4(1 / (tan(fov / 2) * asp), 0, 0, 0,
			0, 1 / tan(fov / 2), 0, 0,
			0, 0, -(fp + bp) / (bp - fp), -1,
			0, 0, -2 * fp * bp / (bp - fp), 0);
	}
};

struct OrtoCamera { 
	vec3 wEye, wLookat, wVup;   

	mat4 V() { 
		vec3 w = normalize(wEye - wLookat);
		vec3 u = normalize(cross(wVup, w));
		vec3 v = cross(w, u);
		return TranslateMatrix(wEye * (-1)) * mat4(u.x, v.x, w.x, 0,
			u.y, v.y, w.y, 0,
			u.z, v.z, w.z, 0,
			0, 0, 0, 1);
	}

	mat4 P() {  //Csala Bálint által tartott konzultáció által tettem tudást eme mátrixra
		return mat4(1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, -2 / 100, 0,
			0, 0, -1, 1);
	}
};

struct Material {
	vec3 kd, ks, ka;
	float shininess;
};


struct Light {
	vec3 La, Le;
	vec4 wLightPos; 
};

float RandomFloat(float a, float b) {  ///forras :https://stackoverflow.com/questions/5289613/generate-random-float-between-two-floats/5289624
	float random = ((float)rand()) / (float)RAND_MAX;
	float diff = b - a;
	float r = random * diff;
	return a + r;
}


class RandomColor : public Texture {
public:
	RandomColor(const int width, const int height) : Texture() {
		std::vector<vec4> image(width * height);
		const vec4 color(RandomFloat(0, 1), RandomFloat(0, 1), RandomFloat(0, 1), 1);
		for (int x = 0; x < width; x++) for (int y = 0; y < height; y++) {
			image[y * width + x] = color;
		}
		create(width, height, image, GL_NEAREST);
	}
};

class SandColor : public Texture {
public:
	SandColor(const int width, const int height) : Texture() {
		std::vector<vec4> image(width * height);
		const vec4 color(0.76f, 0.698f, 0.52f, 1);
		for (int x = 0; x < width; x++) for (int y = 0; y < height; y++) {
			image[y * width + x] = color;
		}
		create(width, height, image, GL_NEAREST);
	}
};



struct RenderState {
	mat4	           MVP, M, Minv, V, P;
	Material* material;
	std::vector<Light> lights;
	Texture* texture;
	vec3	           wEye;
};


class Shader : public GPUProgram {
public:
	virtual void Bind(RenderState state) = 0;

	void setUniformMaterial(const Material& material, const std::string& name) {
		setUniform(material.kd, name + ".kd");
		setUniform(material.ks, name + ".ks");
		setUniform(material.ka, name + ".ka");
		setUniform(material.shininess, name + ".shininess");
	}

	void setUniformLight(const Light& light, const std::string& name) {
		setUniform(light.La, name + ".La");
		setUniform(light.Le, name + ".Le");
		setUniform(light.wLightPos, name + ".wLightPos");
	}
};



class PhongShader : public Shader {
	const char* vertexSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};

		uniform mat4  MVP, M, Minv; // MVP, Model, Model-inverse
		uniform Light[8] lights;    // light sources 
		uniform int   nLights;
		uniform vec3  wEye;         // pos of eye

		layout(location = 0) in vec3  vtxPos;           
		layout(location = 1) in vec3  vtxNorm;      	
		layout(location = 2) in vec2  vtxUV;

		out vec3 wNormal;		    
		out vec3 wView;             
		out vec3 wLight[8];		  
		out vec2 texcoord;
		out vec3 pixel;

		void main() {
			gl_Position = vec4(vtxPos, 1) * MVP; 
			vec4 wPos = vec4(vtxPos, 1) * M;
			for(int i = 0; i < nLights; i++) {
				wLight[i] = lights[i].wLightPos.xyz * wPos.w - wPos.xyz * lights[i].wLightPos.w;
			}
		    wView  = wEye * wPos.w - wPos.xyz;
		    wNormal = (Minv * vec4(vtxNorm, 0)).xyz;
		    texcoord = vtxUV;
			pixel=vtxPos;
		}
	)";


	const char* fragmentSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};

		struct Material {
			vec3 kd, ks, ka;
			float shininess;
		};

		uniform Material material;
		uniform Light[8] lights;    
		uniform int   nLights;
		uniform sampler2D diffuseTexture;

		in  vec3 wNormal;      
		in  vec3 wView;         
		in  vec3 wLight[8];    
		in  vec2 texcoord;
		in  vec3 pixel;
		
        out vec4 fragmentColor; // output goes to frame buffer

		void main() {
			vec3 N = normalize(wNormal);
			vec3 V = normalize(wView); 
			if (dot(N, V) < 0) N = -N;	
			vec3 texColor = texture(diffuseTexture, texcoord).rgb;
			vec3 ka = material.ka * texColor;
			vec3 kd = material.kd * texColor;

			float lepcso = 0.01 / pixel.z;
			lepcso = 1 + floor(pixel.z * 12) / 12;
			vec3 radiance = vec3(0, 0, 0);
			for(int i = 0; i < nLights; i++) {
				vec3 L = normalize(wLight[i]);
				vec3 H = normalize(L + V);
				float cost = max(dot(N,L), 0), cosd = max(dot(N,H), 0);
			
				radiance += ka * lights[i].La + 
										   (lepcso * kd *texColor * cost + material.ks * pow(cosd, material.shininess)) * lights[i].Le*2;
				
				
			}
			fragmentColor = vec4(radiance, 1);
		}
	)";
public:
	PhongShader() { create(vertexSource, fragmentSource, "fragmentColor"); }

	void Bind(RenderState state) {
		Use(); 		
		setUniform(state.MVP, "MVP");
		setUniform(state.M, "M");
		setUniform(state.Minv, "Minv");
		setUniform(state.wEye, "wEye");
		setUniform(*state.texture, std::string("diffuseTexture"));
		setUniformMaterial(*state.material, "material");

		setUniform((int)state.lights.size(), "nLights");
		for (unsigned int i = 0; i < state.lights.size(); i++) {
			setUniformLight(state.lights[i], std::string("lights[") + std::to_string(i) + std::string("]"));
		}
	}
};



class Geometry {
protected:
	unsigned int vao, vbo;       
public:
	Geometry() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
		glGenBuffers(1, &vbo); 
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
	}
	virtual void Draw() = 0;
	virtual void reCreate() = 0;
	~Geometry() {
		glDeleteBuffers(1, &vbo);
		glDeleteVertexArrays(1, &vao);
	}
};

class ParamSurface : public Geometry {
	struct VertexData {
		vec3 position, normal;
		vec2 texcoord;
	};

	unsigned int nVtxPerStrip, nStrips;
public:
	ParamSurface() { nVtxPerStrip = nStrips = 0; }

	virtual void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) = 0;

	VertexData GenVertexData(float u, float v) {
		VertexData vtxData;
		vtxData.texcoord = vec2(u, v);
		Dnum2 X, Y, Z;
		Dnum2 U(u, vec2(1, 0)), V(v, vec2(0, 1));
		eval(U, V, X, Y, Z);
		vtxData.position = vec3(X.f, Y.f, Z.f);
		vec3 drdU(X.d.x, Y.d.x, Z.d.x), drdV(X.d.y, Y.d.y, Z.d.y);
		vtxData.normal = cross(drdU, drdV);
		return vtxData;
	}

	void create(int N = tessellationLevel, int M = tessellationLevel) {
		glBindVertexArray(vao);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		nVtxPerStrip = (M + 1) * 2;
		nStrips = N;
		std::vector<VertexData> vtxData;	
		for (int i = 0; i < N; i++) {
			for (int j = 0; j <= M; j++) {
				vtxData.push_back(GenVertexData((float)j / M, (float)i / N));
				vtxData.push_back(GenVertexData((float)j / M, (float)(i + 1) / N));
			}
		}
		glBufferData(GL_ARRAY_BUFFER, nVtxPerStrip * nStrips * sizeof(VertexData), &vtxData[0], GL_STATIC_DRAW);
		glEnableVertexAttribArray(0); 
		glEnableVertexAttribArray(1);  
		glEnableVertexAttribArray(2);  
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, position));
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, normal));
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, texcoord));
	}
	
	

	void Draw() {
		glBindVertexArray(vao);
		for (unsigned int i = 0; i < nStrips; i++) glDrawArrays(GL_TRIANGLE_STRIP, i * nVtxPerStrip, nVtxPerStrip);
	}

	void reCreate() {
		create(tessellationLevel, tessellationLevel);
	}
};

struct Weight {
	float weight, x, y;
	Weight(float _weight, float _x, float _y) {
		weight = _weight;
		x = _x;
		y = _y;
	}

};

std::vector<Weight> weights;

class RubberSheet : public ParamSurface {
public:
	RubberSheet() { create(); }
	void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) override {
		U = U * 2 - 1;
		V = V * 2 - 1;
		X = U;
		Y = V;
		Z = getZ(X, Y);
	}

	Dnum2 getZ(Dnum2 X, Dnum2 Y) {
		Dnum2 Z = Dnum2(0, vec2(0, 0));
		for (int i = 0; i < weights.size(); i++) {
			Dnum2 dweight = Dnum2(weights[i].weight);
			Dnum2 r0 = Dnum2(0.005);
			Dnum2 dx = Dnum2(weights[i].x, vec2(1, 0));
			Dnum2 dy = Dnum2(weights[i].y, vec2(0, 1));
			Z = Z + ((dweight) / (r0 + Pow(Pow(X - dx, 2) + (Pow(Y - dy, 2)), (0.5f))));
		}
		return Z * (-1);
	}
	
};


class Sphere : public ParamSurface {
public:
	Sphere() { create(); }
	void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) {
		U = U * 2.0f * (float)M_PI, V = V * (float)M_PI;
		X = Cos(U) * Sin(V); Y = Sin(U) * Sin(V); Z = Cos(V);
	}
	void update() {};
};


struct Object {
	Shader* shader;
	Material* material;
	Texture* texture;
	Geometry* geometry;
	vec3 scale, translation, rotationAxis;
	float rotationAngle;
	vec3 velocity;
public:
	Object(Shader* _shader, Material* _material, Texture* _texture, Geometry* _geometry) :
		scale(vec3(1, 1, 1)), translation(vec3(0, 0, 0)), rotationAxis(0, 0, 1), rotationAngle(0) {
		shader = _shader;
		texture = _texture;
		material = _material;
		geometry = _geometry;
		velocity = vec3(0, 0, 0);
	}

	virtual void SetModelingTransform(mat4& M, mat4& Minv) {
		M = ScaleMatrix(scale) * RotationMatrix(rotationAngle, rotationAxis) * TranslateMatrix(translation);
		Minv = TranslateMatrix(-translation) * RotationMatrix(-rotationAngle, rotationAxis) * ScaleMatrix(vec3(1 / scale.x, 1 / scale.y, 1 / scale.z));
	}

	void Draw(RenderState state) {
		mat4 M, Minv;
		SetModelingTransform(M, Minv);
		state.M = M;
		state.Minv = Minv;
		state.MVP = state.M * state.V * state.P;
		state.material = material;
		state.texture = texture;
		shader->Bind(state);
		geometry->Draw();
	}

	virtual void Animate(float tstart, float tend) {
		
		
		for (int i = 0; i < weights.size(); i++) {
			float distx = translation.x - weights[i].x;
			float disty = translation.y - weights[i].y;
			if (sqrtf(powf(distx, 2) + (powf(disty, 2))) < 10*weights[i].weight) {
				
				float vx = (translation.x - weights[i].x)*15;
				float vy = (translation.y - weights[i].y)*15 ;
				if (velocity.x == 0 && velocity.y == 0) {

				}
				else {
					velocity.x -= vx * (tend - tstart);
					velocity.y -= vy * (tend - tstart);
					velocity = normalize(velocity);
				}
				
			} 
		}
		translation = translation + velocity * (tend - tstart);
		if (fabs(translation.x) > 1)
			translation.x *= -1;
		if (fabs(translation.y) > 1)
			translation.y *= -1;
	}

	vec3 returnZ(float u, float v) {
		Dnum2 X, Y;
		Dnum2 U(u, vec2(1, 0)), V(v, vec2(0, 1));
		U = U ;
		V = V ;
		X = U;
		Y = V;

		Dnum2 Z = Dnum2(0, vec2(0, 0));
		for (int i = 0; i < weights.size(); i++) {
			Dnum2 dweight = Dnum2(weights[i].weight);
			Dnum2 dx = Dnum2(weights[i].x, vec2(1, 0));
			Dnum2 dy = Dnum2(weights[i].y, vec2(0, 1));
			Dnum2 r0 = Dnum2(0.005);
			Z = Z + (dweight / (r0 + Pow(Pow(X - dx, 2) + (Pow(Y - dy, 2)), (0.5f))));
		}
		Z = Z * (-1);
		vec3 eltolt(X.f, Y.f, Z.f);
		return eltolt;
	}

};


vec4 qmul(vec4 q1, vec4 q2) {
	vec4 q;
	q.x = (q1.w * q2.x) + (q1.x * q2.w) + (q1.y * q2.z) - (q1.z * q2.y);
	q.y = (q1.w * q2.y) - (q1.x * q2.z) + (q1.y * q2.w) + (q1.z * q2.x);
	q.z = (q1.w * q2.z) + (q1.x * q2.y) - (q1.y * q2.x) + (q1.z * q2.w);
	q.w = (q1.w * q2.w) - (q1.x * q2.x) - (q1.y * q2.y) - (q1.z * q2.z);
	return q;
}

vec4 rotate(vec4 u, vec4 q) {
	vec4 q_inv = vec4(-q.x, -q.y, -q.z, q.w);
	q = qmul(qmul(q, vec4(u.x, u.y, u.z, u.w)), q_inv);
	return vec4(q.x, q.y, q.z, q.w);
}



class Scene {

	OrtoCamera camera;
	Camera camera2;
	bool felulkamera;
	std::vector<Light> lights;
public:
	std::vector<Object*> objects;
	Object* sheetObject;
	Texture* masikszin;
	vec4 kezdokamera0 = vec4(2.5, -0.5, 5, 10);
	vec4 kezdokamera1 = vec4(-0.5, 2.5, 5, 10);
	

	void Build() {
		felulkamera = true;

		Shader* phongShader = new PhongShader();

		Material* material0 = new Material;
		material0->kd = vec3(0.6f, 0.4f, 0.2f);
		material0->ks = vec3(4, 4, 4);
		material0->ka = vec3(0.1f, 0.1f, 0.1f);
		material0->shininess = 100;

		Material* material1 = new Material;
		material1->kd = vec3(0.8f, 0.6f, 0.4f);
		material1->ks = vec3(0.3f, 0.3f, 0.3f);
		material1->ka = vec3(0.2f, 0.2f, 0.2f);
		material1->shininess = 300;


		Texture* egyikszin = new SandColor(1, 1);
		masikszin = new RandomColor(1, 1);

		Geometry* sphere = new Sphere();
		Geometry* sheet = new RubberSheet();

		Object* sphereObject1 = new Object(phongShader, material0, masikszin, sphere);
		sphereObject1->translation = vec3(-0.8, -0.8, 0.1);
		sphereObject1->scale = vec3(0.1, 0.1, 0.1);
		objects.push_back(sphereObject1);

		Object* sheetObj = new Object(phongShader, material0, egyikszin, sheet);
		sheetObj->translation = vec3(0, 0, 0);
		sheetObj->scale = vec3(1, 1, 1);
		sheetObject = sheetObj;

		int nObjects = objects.size();

		camera.wEye = vec3(0, 0, 8);
		camera.wLookat = vec3(0, 0, 0);
		camera.wVup = vec3(0, 1, 0);

		lights.resize(2);
		lights[0].wLightPos = vec4(2.5, -0.5, 5, 10);	
		lights[0].La = vec3(0.1f, 0.1f, 0.4f);
		lights[0].Le = vec3(0.3f, 0.3f, 0.3f);

		lights[1].wLightPos = vec4(-0.5, 2.5, 5, 10);	
		lights[1].La = vec3(0.2f, 0.2f, 0.2f);
		lights[1].Le = vec3(0.4f, 0.4f, 0.4f);


	}

	void Render() {
		RenderState state;
		if (felulkamera) {
			state.wEye = camera.wEye;
			state.V = camera.V();
			state.P = camera.P();
			state.lights = lights;
		}
		else {
			camera2.wEye = objects.back()->translation;
			if (objects.back()->velocity.x == 0 && objects.back()->velocity.y == 0) {
				camera2.wLookat = objects.back()->translation + vec3(0.1, 0.1, 0);
			}
			else {
				camera2.wLookat = objects.back()->translation + objects.back()->velocity;
			}
			camera2.wVup = vec3(0, 0, 1);
			state.wEye = camera2.wEye;
			state.V = camera2.V();
			state.P = camera2.P();
			state.lights = lights;
		}


		for (Object* obj : objects) obj->Draw(state);
		sheetObject->Draw(state);
	}

	void Animate(float tstart, float tend) {
		for (int i = 0; i < weights.size(); i++) {
			int o = 0;
			for (Object* obj : objects) {
				

				if (powf(powf(obj->translation.x - weights[i].x, 2) + (powf(obj->translation.y - weights[i].y, 2)), (0.5f)) < 0.02) {
					objects.erase(objects.begin()+o);
				}

				o++;
			}
			
		}
		for (Object* obj : objects) {
			obj->Animate(tstart, tend);
			obj->translation.z = sheetObject->returnZ(obj->translation.x , obj->translation.y ).z +0.1;
		}
		vec4 q(cos(tend / 4), sin(tend / 4) * cos(tend) / 2, sin(tend / 4) * sin(tend) / 2, sin(tend / 4) * sqrtf(3 / 4));
		lights[0].wLightPos = kezdokamera1 + rotate(kezdokamera0 - kezdokamera1, q);
		lights[1].wLightPos = kezdokamera0 + rotate(kezdokamera1 - kezdokamera0, q);

	}

	void ujLabda(float cX, float cY) {
		Texture* texture15x20 = new RandomColor(1, 1);
		Geometry* sphere = new Sphere();
		Material* material0 = new Material;
		material0->kd = vec3(0.6f, 1.0f, 0.2f);
		material0->ks = vec3(4, 4, 4);
		material0->ka = vec3(0.1f, 0.1f, 0.1f);
		material0->shininess = 100;

		Object* sphereObject1 = new Object(new PhongShader, material0, masikszin, sphere);
		sphereObject1->translation = vec3(-0.8, -0.8, 0.1);
		sphereObject1->scale = vec3(0.1, 0.1, 0.1);
		sphereObject1->velocity = vec3((cX + 0.8), (cY + 0.8), 0);
		RandomColor* ujszin = new RandomColor(1, 1);
		objects[0]->texture = ujszin;
		masikszin = ujszin;
		objects.push_back(sphereObject1);
	}

	void labdaKam() {
		felulkamera = false;
	}

	void fentiKam() {
		felulkamera = true;
	}
	bool getKam() {
		return felulkamera;
	}

	void updateSheet() {
		sheetObject->geometry->reCreate();
	}
};

Scene scene;


void onInitialization() {

	printf("A vegeredmeny %f", result);

	glViewport(0, 0, windowWidth, windowHeight);
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	scene.Build();
}


void onDisplay() {
	glClearColor(0.5f, 0.5f, 0.8f, 1.0f);							
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); 
	scene.Render();
	glutSwapBuffers();									
}

void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == ' ') {
		if (scene.getKam() == true) {
			scene.labdaKam();
		}
		else {
			scene.fentiKam();
		}
		
	}
	glutPostRedisplay();

}

void onKeyboardUp(unsigned char key, int pX, int pY) { 
	
}


void onMouse(int button, int state, int pX, int pY) { 
	float cX = 2.0f * pX / windowWidth - 1;	
	float cY = 1.0f - 2.0f * pY / windowHeight;

	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN)
	{
		scene.ujLabda(cX, cY);
	}

	if (button == GLUT_RIGHT_BUTTON && state == GLUT_DOWN) {
		weights.push_back(Weight(0.02+0.02*weights.size(), cX, cY));
		scene.updateSheet();
	}

}

void onMouseMotion(int pX, int pY) {
}

void onIdle() {
	static float tend = 0;
	const float dt = 0.1f; 
	float tstart = tend;
	tend = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;

	for (float t = tstart; t < tend; t += dt) {
		float Dt = fmin(dt, tend - t);
		scene.Animate(t, t + Dt);
	}
	glutPostRedisplay();
}
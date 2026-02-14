#include <SDL2/SDL.h>
#include <GL/glew.h>
#include <GL/gl.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <xmmintrin.h>

#define PI                  acos(-1.0)

#define WIN_W               1200
#define WIN_H               700
#define WORLD_X             8.0f
#define WORLD_Y             8.0f
#define WORLD_Z             8.0f

#define GLOBAL_DAMPING      0.15f

#define SLEEP_VEL_EPS       0.02f
#define SLEEP_FRAMES        30

#define MAX_NODES           1024
#define MAX_BODIES          40
#define MAX_EDGES           2048
#define MAX_FACES           2048
#define NODES_PER_BODY      50
#define EDGES_PER_BODY      1000
#define ADJ_MAX             14

#define EDGEKEY_EMPTY   0xFFFFFFFFu
#define EDGEKEY_TOMB    0xFFFFFFFEu
#define EDGEHASH_CAP    2048
#define EDGEHASH_MASK   (EDGEHASH_CAP - 1)

#define NODEKEY_EMPTY   0xFFFFFFFFu
#define NODEKEY_TOMB    0xFFFFFFFEu
#define NODEHASH_CAP    512
#define NODEHASH_MASK   (NODEHASH_CAP -1)

#define FACEKEY_EMPTY   0xFFFFFFFFu
#define FACEKEY_TOMB    0xFFFFFFFEu
#define FACEHASH_CAP    2048
#define FACEHASH_MASK   (FACEHASH_CAP - 1)

static const int8_t OFFS[12][3] = {
    {+1,+1, 0}, {+1,-1, 0}, {-1,+1, 0}, {-1,-1, 0},
    {+1, 0,+1}, {+1, 0,-1}, {-1, 0,+1}, {-1, 0,-1},
    { 0,+1,+1}, { 0,+1,-1}, { 0,-1,+1}, { 0,-1,-1},
};

#define REST_LEN            0.5f

#define MAX_PARTICLES       4096

#define DISC_SEG            24
#define MAX_DRAW_EDGES      (MAX_BODIES * EDGES_PER_BODY)
#define MAX_DRAW_NODES      (MAX_BODIES * NODES_PER_BODY)

static int gBodyCount = 0;

typedef struct { float x, y, z; } Vec3;
typedef struct { int x, y, z; } Vec3Int;

typedef enum {
    EDGE_NONE   = 0,
    EDGE_TENDON,
    EDGE_BONE,
    EDGE_MUSCLE
} EdgeType;

typedef enum {
    NODE_NORMAL = 0,
    NODE_HEART  = 1,
    NODE_WEAPON = 2
} NodeType;

typedef struct {
    Vec3 pos, vel;
    float life;
    uint8_t r,g,b,a;
} Particle;
static Particle gP[MAX_PARTICLES];
static int      gPCount = 0;
static float    gP_Pos3[MAX_PARTICLES * 3];
static uint8_t  gP_Col4[MAX_PARTICLES * 4];

typedef struct {
    uint8_t r,g,b,a;
    uint8_t count;
    float bounceE;
} HitFX;

typedef struct {
    int   linearity; //Number of neighboring nodes each node optimally wants to attach to
    float linearityBias; //Multiplier for linearity variance; how 'picky' linearity is
    float skeletal; //Pre-edge-choice bias to form bone edges when neighboring another bone edge; 0 means normal distribution
    float boneP; float muscleP; float tendonP; //Normal probabilities of each edge type generating; when setting, ensure sums to 1
} Genetics;

typedef struct {
    Vec3  pos;              // 3D position (simulation space)
    Vec3  vel;              // 3D velocity
    Vec3  force;            // accumulator each tick
    float invMass;          // 0 = fixed, else 1/mass
    uint16_t owner;         // organism id
    NodeType type;
    float radius;
    uint8_t alive;
    uint8_t distHeart;      // number of nodes in the shortest path to the heart node (heart is 1, disconnected is 0)
    Vec3Int lattice;        // skeleton lattice coordinates
} Node;
static float    gNodePos_Type[3][MAX_DRAW_NODES * 18];
static float    gNodeUV[MAX_DRAW_NODES * 12];
static uint8_t  gNodeCol_Type[3][MAX_DRAW_NODES * 24];
static int      gNodeCountType[3];
 
static GLuint gNodeSpriteTexId = 0;

typedef struct {
    int16_t activeSlot;
    int a, b;               // node indices
    float restLen;          // spring rest length
    float k;                // spring stiffness
    float damping;          // edge damping
    EdgeType type;          // edge type
    float muscleTargetMul;  // ratio of relaxed length to contract to
    float muscleSignal;     // signal input to trigger
    int hp;
} Edge;
static float    gEdgePos_Type[4][MAX_DRAW_EDGES * 18];
static int      gEdgeCount_Type[4];

typedef struct {
    uint16_t    a, b, c;
    uint16_t    inside;
    uint8_t     alive;
} Face;

typedef struct {
    Node nodes[NODES_PER_BODY]; 
    int nodeCount;
    Edge edges[EDGES_PER_BODY]; 
    int edgeCount;
    uint16_t activeEdgeIdx[EDGES_PER_BODY];
    uint16_t activeEdgeCount;
    Face frontier[MAX_FACES];
    int faceCount;
    uint8_t adjCount[NODES_PER_BODY];
    uint16_t adjNode[NODES_PER_BODY][ADJ_MAX];
    uint16_t adjEdge[NODES_PER_BODY][ADJ_MAX];

    uint32_t edgeHashKey[EDGEHASH_CAP];
    uint16_t edgeHashVal[EDGEHASH_CAP];

    uint32_t nodeHashKey[NODEHASH_CAP];
    uint32_t nodeHashVal[NODEHASH_CAP];

    uint32_t faceHashKey[FACEHASH_CAP];
    uint16_t faceHashVal[FACEHASH_CAP];

    uint16_t jointCandIdx[NODES_PER_BODY];
    uint16_t jointCandCount;
    int16_t jointCandSlot[NODES_PER_BODY];

    uint16_t id;

    uint8_t sleepFrames;
    uint8_t sleeping;

    uint8_t topoDirty;
    uint8_t boneDeg[NODES_PER_BODY];

    Vec3 bsCenter;
    float bsRadius;
    Vec3 bbMin, bbMax;

    Genetics genes;
} Body;
static Body gBodies[MAX_BODIES];

typedef struct {
    Vec3 center;            // world-space center to look at
    float scale;            // pixels per world unit
    float yaw;              // rotation around Z
    float pitch;            // rotation around X
    float focal;            // perspective strength / focal length
    float nearZ;            // near plane in camera space (positive)
    float cy, sy, cp, sp;   // Camera transcendenal storage
} Camera;
static Vec3 gCamNode[MAX_BODIES][NODES_PER_BODY];
static float gCamTrim[MAX_BODIES][NODES_PER_BODY];
static float gProjKx = 1.0f;
static float gProjKy = 1.0f;

static const uint8_t EDGE_COL[4][4] = {
    {200,200,200,255}, //NONE
    {120,200,255,255}, //TENDON
    {230,230,210,255}, //BONE
    {255,120,120,255}, //MUSCLE
};

static const uint8_t NODE_COL[3][4] = {
    { 80, 80,255,255}, //NORMAL
    {255, 50, 50,255}, //HEART
    {255,200, 50,255}, //WEAPON
};

static const HitFX FX_ON_BREAK[4] = {
    {255,255,255,255,  0, 0.0f}, //NONE
    {120,200,255,255, 18, 0.0f}, //TENDON
    {230,230,210,255, 30, 0.6f}, //BONE
    {255,120,120,255, 22, 0.0f}, //MUSCLE
};

static inline float frand01(void) {
    return (float)rand() / (float)RAND_MAX;
}
static inline float frandRange(float lo, float hi) {
    return lo + (hi - lo) * frand01();
}
static inline float irandRange(int lo, int hi) {
    return lo + (rand() % (hi - lo + 1));
}

static inline float sqr(float a) {return a * a;}
static inline float invSqrt(float a, int precision) { 
    if (a <= 0.0f) return 0.0f;
    float b = _mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(a)));
    for (int i=0; i<precision; ++i) {
        b *= 1.5 - 0.5 * a * sqr(b);
    }
    return(b);
}

static inline Vec3 v3(float x, float y, float z) { Vec3 v = {x,y,z}; return v; }
static inline Vec3 v3Add(Vec3 a, Vec3 b) { return v3(a.x+b.x, a.y+b.y, a.z+b.z); }
static inline Vec3 v3Sub(Vec3 a, Vec3 b) { return v3(a.x-b.x, a.y-b.y, a.z-b.z); }
static inline Vec3 v3Scale(Vec3 a, float s) { return v3(a.x*s, a.y*s, a.z*s); }
static inline Vec3 v3Rand(void) { return v3(frand01()*2-1, frand01()*2-1, frand01()*2-1);}
static inline float v3Dot(Vec3 a, Vec3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
static inline float v3DotSq(Vec3 a) { return a.x*a.x + a.y*a.y + a.z*a.z; }
static inline Vec3 v3Normalize(Vec3 a, int sqrtIter, float precision, int failureMode) {
    float len2 = v3DotSq(a);
    if (len2 < precision) {
        if (failureMode) len2 = precision;
        else return v3(0, 0, 0);
    }
    float invLen = invSqrt(len2, sqrtIter);
    return(v3Scale(a, invLen));
}
static inline float v3Len(Vec3 a) { return sqrtf(v3DotSq(a)); }
static inline float v3Dist(Vec3 a, Vec3 b) { return v3Len(v3Sub(b, a));}
static inline char v3Tol(Vec3 a, float b) { return v3DotSq(a) < sqr(b);}
static inline char v3FullMax(Vec3 a, Vec3 b) { return (a.x > b.x && a.y > b.y && a.z > b.z)? 1 : 0;}
static inline void v3Zero(Vec3* v) { v->x = v->y = v->z = 0.0f; }

static inline Vec3Int v3Int(int x, int y, int z) { Vec3Int v = {x,y,z}; return v;}
static inline Vec3Int v3IntAdd(Vec3Int a, Vec3Int b) {return v3Int(a.x+b.x, a.y+b.y, a.z+b.z);}
static inline Vec3Int v3IntSub(Vec3Int a, Vec3Int b) {return v3Int(a.x-b.x, a.y-b.y, a.z-b.z);}
static inline Vec3Int v3IntScale(Vec3Int a, int b) {return v3Int(a.x*b, a.y*b, a.z*b);}
static inline int v3IntDotSq(Vec3Int a) {return (int)a.x*a.x + (int)a.y*a.y + (int)a.z*a.z;}
static inline int v3IntDist2(Vec3Int a, Vec3Int b) { return v3IntDotSq(v3Int(b.x-a.x, b.y-a.y, b.z-a.z));}
static inline Vec3 v3IntTov3(Vec3Int a) {return v3((float)a.x, (float)a.y, (float)a.z);}


#define CLAMP(v, l, h) ((v) < (l) ? (l) : ((v) > (h) ? (h) : (v)))
#define CLAMP01(v) ((v) < (0) ? (0) : ((v) > 1 ? (1) : (v)))

static inline uint32_t hash32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}
static inline uint32_t edgeKey(int a, int b){
    if (a > b) { int t=a; a=b; b=t; }
    return ((uint32_t)a << 16) | (uint32_t)b;
}
static inline uint32_t edgeHash0(uint32_t key){return hash32(key) & EDGEHASH_MASK;}

static inline uint32_t packCoord(Vec3Int coord) {
    uint32_t x = (uint32_t)(coord.x + 512) & 1023u;
    uint32_t y = (uint32_t)(coord.y + 512) & 1023u;
    uint32_t z = (uint32_t)(coord.z + 512) & 1023u;
    return (x << 20) | (y << 10) | z;
}
static inline uint32_t nodeHash0(uint32_t key){return hash32(key) & NODEHASH_MASK;}

static inline void sort3u16(uint16_t *a, uint16_t *b, uint16_t *c) {
    uint16_t x=*a,y=*b,z=*c,t;
    if (x>y) {t=x;x=y;y=t;}
    if (y>z) {t=y;y=z;z=t;}
    if (x>y) {t=x;x=y;y=t;}
    *a=x;*b=y;*c=z;
}
static inline uint32_t faceKey(uint16_t a, uint16_t b, uint16_t c) {
    sort3u16(&a,&b,&c);
    return ((uint32_t)a <<20) | ((uint32_t)b << 10) | (uint32_t)c;
}
static inline uint32_t faceHash0(uint32_t key){ return hash32(key) & FACEHASH_MASK;}

static inline void wakeBody(Body *b) { b->sleeping = 0; b->sleepFrames = 0; }
static int  bodyIsQuiet(const Body *b) {
    float maxV2 = 0.0f;
    for (int i = 0; i < b->nodeCount; ++i) {
        Vec3 v = b->nodes[i].vel;
        float v2 = sqr(v.x) + sqr(v.y) + sqr(v.z);
        if (v2 > maxV2) maxV2 = v2;
    }
    return maxV2 < (SLEEP_VEL_EPS * SLEEP_VEL_EPS);
}
static void bodyBounds(Body *b) {
    char foundLive = 0;
    Vec3 min, max = v3(0,0,0);
    for (int i=0; i<b->nodeCount; i++) {
        Node *node = &b->nodes[i]; if (!node->alive) continue;
        Vec3 pos = node->pos;
        if (foundLive == 0) {
            min = max = pos;
            foundLive = 1;
        } else {
            min.x = fmin(min.x, pos.x); min.y = fmin(min.y, pos.y); min.z = fmin(min.z, pos.z);
            max.x = fmax(max.x, pos.x); max.y = fmax(max.y, pos.y); max.z = fmax(max.z, pos.z);
        }
    }
    if (!foundLive) {b->bsRadius = 0.0f; b->bsCenter = v3(0,0,0); b->bbMin = v3(0,0,0); b->bbMax = v3(0,0,0); return;}
    b->bsCenter = v3Scale(v3Add(min, max), 0.5);
    b->bbMin = min;
    b->bbMax = max;
    float rMax = 0.0f;
    for (int i=0; i<b->nodeCount; i++) {
        Node *node = &b->nodes[i]; if (!node->alive) continue;
        float r = v3Dist(node->pos, b->bsCenter) + node->radius;
        if (rMax < r) rMax = r;
    }
    b->bsRadius = rMax;
}



static void initNodeSpriteTexture(void) {
    const int W = 64, H = 64;
    static uint8_t pix[64 * 64 * 4];
    const float cx = 0.5f, cy = 0.5f;
    const float hx = 0.35f, hy = 0.35f;
    const float edgeSoft = 0.06f;
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float u = (x + 0.5f) / (float)W, v = (y + 0.5f) / (float)H;
            float dx = u - cx, dy = v - cy;
            float r  = sqrtf(dx*dx + dy*dy);
            float rn = r / 0.5f;
            float a = 1.0f - CLAMP01((rn - (1.0f - edgeSoft)) / edgeSoft); if (rn >= 1.0f) a = 0.0f;
            float dhx = u - hx, dhy = v - hy;
            float h2  = dhx*dhx + dhy*dhy;
            float hi  = expf(-h2 * 80.0f);
            float shade = CLAMP01(1.0f - 0.55f * CLAMP01(rn) + 0.35f * hi);
            uint8_t I = (uint8_t)(shade * 255.0f), A = (uint8_t)(a * 255.0f);
            int i = (y * W + x) * 4;
            pix[i+0] = I; pix[i+1] = I; pix[i+2] = I; pix[i+3] = A;
        }
    }
    glGenTextures(1, &gNodeSpriteTexId);
    glBindTexture(GL_TEXTURE_2D, gNodeSpriteTexId);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, pix);
}
static void initNodeUVConst(void) {
    static const float UV[12] = {0,0, 1,0, 1,1, 0,0, 1,1, 0,1};
    for (int i=0; i<MAX_DRAW_NODES; ++i) {
        memcpy(&gNodeUV[i * 12], UV, sizeof(UV));
    }
}

static inline void cameraTranscendentals(Camera *cam) {
    cam->cy = cosf(cam->yaw);   
    cam->sy = sinf(cam->yaw);
    cam->cp = cosf(cam->pitch); 
    cam->sp = sinf(cam->pitch);
}
static inline Vec3 cameraEye(const Camera *cam, Vec3 p) {
    Vec3 r = v3Sub(p, cam->center);
    float y1 = -cam->sy * r.x + cam->cy * r.y;
    return v3(cam->cy * r.x + cam->sy * r.y, cam->cp * y1 - cam->sp * r.z, -(cam->sp * y1 + cam->cp * r.z + cam->focal));
}
static void buildCameraNodes(const Camera *cam) {
    for (int bi=0; bi<gBodyCount; ++bi) {
        Body *b = &gBodies[bi];
        for (int ni=0; ni<b->nodeCount; ++ni) {
            Vec3 cpos = cameraEye(cam, b->nodes[ni].pos);
            gCamNode[bi][ni] = cpos;

            float L2 = v3DotSq(cpos);
            gCamTrim[bi][ni] = CLAMP((L2 > 1e-12f) ? invSqrt(L2, 1) * -cpos.z * b->nodes[ni].radius : 0.0f, 0.05f, 1.0f);
        }
    }
}

static void drawEdges(const Camera *cam) {
    gEdgeCount_Type[1] = gEdgeCount_Type[2] = gEdgeCount_Type[3] = 0;
    float tN = 0.0025f;
    float jKx = gProjKx, jKy = gProjKy, invJKx = tN / jKx, invJKy = tN / jKy;
    for (int bi=0; bi<gBodyCount; ++bi) {
        const Body *b = &gBodies[bi];
        for (int ei=0; ei < b->edgeCount; ++ei) {
            const Edge *e = &b->edges[ei];
            uint8_t t = (uint8_t)e->type; if (t == EDGE_NONE) continue;
            if (gEdgeCount_Type[t] >= MAX_DRAW_EDGES) continue;
            Vec3 Ac = gCamNode[bi][e->a]; Vec3 Bc = gCamNode[bi][e->b]; if (Ac.z > -1e-3f || Bc.z > -1e-3f) continue;
            Vec3 ABc = v3Sub(Bc, Ac); float len2 = v3DotSq(ABc); if (len2 < 1e-12f) continue; float invLen = invSqrt(len2, 1); Vec3 dir3 = v3Scale(ABc, invLen);
            float raTrim = gCamTrim[bi][e->a], rbTrim = gCamTrim[bi][e->b];
            if (len2 * invLen - (raTrim + rbTrim) <= 1e-6f) continue;
            Vec3 A = v3Add(Ac, v3Scale(dir3, raTrim)), B = v3Sub(Bc, v3Scale(dir3, rbTrim));
            float Az = -A.z, Bz = -B.z, invAz = 1.0f / Az, invBz = 1.0f / Bz;
            float dx = jKx * (B.x * invBz - A.x * invAz), dy = jKy * (A.y * invAz - B.y * invBz);
            float d2 = sqr(dx) + sqr(dy); if (d2 < 1e-12f) continue; float invD = invSqrt(d2, 0);
            float px = dy * invD * invJKx, py = dx * invD * invJKy;
            Vec3 pA = v3(px * Az, py * Az, 0.0f), pB = v3(px * Bz, py * Bz, 0.0f);
            Vec3 A0 = v3Sub(A, pA); Vec3 A1 = v3Add(A, pA); Vec3 B1 = v3Add(B, pB); Vec3 B0 = v3Sub(B, pB);
            Vec3 V[6] = {A0, A1, B1, A0, B1, B0};
            float *pos = &gEdgePos_Type[t][gEdgeCount_Type[t] * 18];
            for (int ti=0; ti<6; ++ti) {*pos++ = V[ti].x; *pos++ = V[ti].y; *pos++ = V[ti].z;}
            gEdgeCount_Type[t]++;
        }
    }

    DRAW:
        glEnable(GL_DEPTH_TEST);
        glDepthMask(GL_TRUE);
        glEnableClientState(GL_VERTEX_ARRAY);
        for (int t=1; t<=3; ++t) {
            int ec = gEdgeCount_Type[t];
            if (!ec) continue;
            const uint8_t *c = EDGE_COL[t];
            glColor4ub(c[0],c[1],c[2],c[3]);
            glVertexPointer(3, GL_FLOAT, 0, gEdgePos_Type[t]);
            glDrawArrays(GL_TRIANGLES, 0, ec * 6);
        }
        glDisableClientState(GL_VERTEX_ARRAY);
}
static void drawNodes(const Camera *cam) {
    gNodeCountType[0]=gNodeCountType[1]=gNodeCountType[2]=0;
    for (int bi=0; bi<gBodyCount; ++bi) {
        Body *b = &gBodies[bi];
        for (int ni = 0; ni < b->nodeCount; ++ni) {
            Node *n = &b->nodes[ni]; if (!n->alive) continue;   //Get node; fallback if not alive
            uint8_t t = (uint8_t)n->type; if (gNodeCountType[t] >= MAX_DRAW_NODES) continue;    //Get type; fallback if type array is full
            Vec3 C = gCamNode[bi][ni]; if (C.z > -1e-3f) continue;  //Camera persp; fallback if close to or behind camera
            float *pos = &gNodePos_Type[t][gNodeCountType[t] * 18]; //generate triangles vertex postion array
            float R = n->radius, cxm = C.x - R, cxp = C.x + R, cym = C.y - R, cyp = C.y + R, cz = C.z; //Precomputations
            pos[ 0]=cxm; pos[ 1]=cym; pos[ 2]=cz; pos[ 3]=cxp; pos[ 4]=cym; pos[ 5]=cz; pos[ 6]=cxp; pos[ 7]=cyp; pos[ 8]=cz;   //first triangle positons
            pos[ 9]=cxm; pos[10]=cym; pos[11]=cz; pos[12]=cxp; pos[13]=cyp; pos[14]=cz; pos[15]=cxm; pos[16]=cyp; pos[17]=cz;   //second triangle positions
            const uint8_t *c = NODE_COL[t];     //color
            uint8_t *col = &gNodeCol_Type[t][gNodeCountType[t] * 24];   //color array
            for (int i=0; i<6; ++i) { *col++=c[0]; *col++=c[1]; *col++=c[2]; *col++=c[3]; } //color array populate
            gNodeCountType[t]++;        //iterate node count
        }
    }
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);

    glDisable(GL_BLEND);
    glEnable(GL_ALPHA_TEST);
    glAlphaFunc(GL_GREATER, 0.1f);

    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, gNodeSpriteTexId);
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);

    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_TEXTURE_COORD_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);

    glTexCoordPointer(2, GL_FLOAT, 0, gNodeUV);

    for (int t=0; t<=2; ++t) {
        int nc = gNodeCountType[t];
        if (!nc) continue;

        glVertexPointer(3, GL_FLOAT, 0, gNodePos_Type[t]);
        glColorPointer(4, GL_UNSIGNED_BYTE, 0, gNodeCol_Type[t]);
        glDrawArrays(GL_TRIANGLES, 0, nc * 6);
    }

    glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_TEXTURE_COORD_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);

    glDisable(GL_TEXTURE_2D);
    glDisable(GL_ALPHA_TEST);
    glEnable(GL_BLEND);
}

static void spawnBreakFX(Vec3 p, Vec3 n, int count, uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
    for (int i=0; i<count; i++) {
        if (gPCount >= MAX_PARTICLES) break;
        Particle *q = &gP[gPCount++];
        float s = 1.0f + frand01()*3.0f;
        Vec3 randv = v3(frand01() * 2 - 1, frand01() * 2 - 1, frand01() * 2 - 1);
        q->pos = p;
        q->vel = v3Add(v3Scale(n, s*2.0f), v3Scale(randv, s));
        q->life = 0.75f + frand01()*0.35f;
        q->r=r; q->g=g; q->b=b; q->a=a;
    }
}
static void updateParticles(float dt) {
    int w = 0;
    for (int i=0; i<gPCount; i++) {
        Particle *p = &gP[i];
        p->life -= dt;
        if (p->life <= 0.0f) continue;
        p->vel = v3Scale(p->vel, 0.98f);
        p->pos = v3Add(p->pos, v3Scale(p->vel, dt));
        p->a = (uint8_t)(255.0f * CLAMP01(p->life));
        if (w != i) gP[w] = *p;
        w++;
    }
    gPCount = w;
}
static void drawParticles(const Camera *cam) {
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);
    glPointSize(3.0f);

    for (int i = 0; i < gPCount; ++i) {
        Vec3 C = cameraEye(cam, gP[i].pos);
        int pi = i * 3;
        gP_Pos3[pi + 0] = C.x;
        gP_Pos3[pi + 1] = C.y;
        gP_Pos3[pi + 2] = C.z;
        int ci = i * 4;
        gP_Col4[ci + 0] = gP[i].r;
        gP_Col4[ci + 1] = gP[i].g;
        gP_Col4[ci + 2] = gP[i].b;
        gP_Col4[ci + 3] = gP[i].a;
    }
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
    glVertexPointer(3, GL_FLOAT, 0, gP_Pos3);
    glColorPointer(4, GL_UNSIGNED_BYTE, 0, gP_Col4);
    glDrawArrays(GL_POINTS, 0, gPCount);
    glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);
    glDepthMask(GL_TRUE);
}

static void setPerspective(SDL_Window *win) {
    int w,h; SDL_GL_GetDrawableSize(win,&w,&h);
    glViewport(0, 0, w, h);

    float aspect = (float)w / (float)h;
    float fovy = 60.0f;
    float zNear = 0.1f;
    float zFar  = 100.0f;

    float f = 1.0f / tanf((fovy * 0.5f) * (float)PI / 180.0f);
    gProjKy = f;
    gProjKx = f / aspect;
    float M[16] = {
        gProjKx, 0, 0, 0,
        0, f, 0, 0,
        0, 0, (zFar+zNear)/(zNear-zFar), -1,
        0, 0, (2*zFar*zNear)/(zNear-zFar), 0
    };
    glMatrixMode(GL_PROJECTION);
    glLoadMatrixf(M);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

static int initSDL(SDL_Window **outWin, SDL_GLContext *outGL) {
    SDL_SetHint(SDL_HINT_VIDEO_X11_NET_WM_BYPASS_COMPOSITOR, "0");
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        printf("SDL_Init error: %s\n", SDL_GetError());
        return 0;
    }

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_COMPATIBILITY);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 2);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 1);

    SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, 1);
    SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, 4);

    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
    SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8);

    SDL_GL_SetAttribute(SDL_GL_RED_SIZE,   8);
    SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE,  8);
    SDL_GL_SetAttribute(SDL_GL_ALPHA_SIZE, 8);

    SDL_Window *win = SDL_CreateWindow(
        "NodeSim Rev-0",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        WIN_W, WIN_H,
        SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN
    );
    if (!win) {
        printf("SDL_CreateWindow error: %s\n", SDL_GetError());
        SDL_Quit();
        return 0;
    }

    SDL_GLContext glctx = SDL_GL_CreateContext(win);
    if (!glctx) {
        printf("SDL_CreateRenderer error: %s\n", SDL_GetError());
        SDL_DestroyWindow(win);
        SDL_Quit();
        return 0;
    }
    SDL_GL_MakeCurrent(win, glctx);

    SDL_GL_SetSwapInterval(1);

    glewExperimental = GL_TRUE;
    GLenum err = glewInit();
    if (err != GLEW_OK) {
        printf("glewInit error: %s\n", (const char*)glewGetErrorString(err));
        return 0;
    }
    
    glGetError();

    initNodeSpriteTexture();

    glViewport(0, 0, WIN_W, WIN_H);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glDisable(GL_CULL_FACE);
    glEnable(GL_BLEND);
    glEnable(GL_MULTISAMPLE);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glClearColor(0.04f, 0.04f, 0.05f, 1.0f);
    glMatrixMode(GL_PROJECTION);
    glClearDepth(1.0);

    *outWin = win;
    *outGL  = glctx;

    return 1;
}
static void shutdownSDL(SDL_Window *win, SDL_GLContext gl) {
    if (gl) SDL_GL_DeleteContext(gl);
    if (win) SDL_DestroyWindow(win);
    SDL_Quit();
}



static int edgeHashFindSlot(const Body *b, uint32_t key) {
    uint32_t h = edgeHash0(key);
    for (int i=0; i<EDGEHASH_CAP; ++i) {
        uint32_t k = b->edgeHashKey[h];
        if (k == EDGEKEY_EMPTY) return -1;
        if (k == key) return (int)h;
        h = (h + 1) & EDGEHASH_MASK;
    }
    return -1;
}
static int edgeHashFindInsertSlot(const Body *b, uint32_t key) {
    uint32_t h = edgeHash0(key);
    int firstTomb = -1;
    for (int i=0; i<EDGEHASH_CAP; ++i) {
        uint32_t k = b->edgeHashKey[h];
        if (k == EDGEKEY_EMPTY) return (firstTomb >= 0) ? firstTomb : (int)h;
        if (k == EDGEKEY_TOMB && firstTomb < 0) firstTomb = (int)h;
        if (k == key) return (int)h;
        h = (h + 1) & EDGEHASH_MASK;
    }
    return firstTomb;
}
static inline void activeRemoveEdge(Body *b, uint16_t ei) {
    int16_t slot = b->edges[ei].activeSlot;
    if (slot < 0 || b->activeEdgeCount == 0) return;
    int16_t lastEi = b->activeEdgeIdx[--b->activeEdgeCount];
    if ((uint16_t)slot != b->activeEdgeCount) {
        b->activeEdgeIdx[slot] = lastEi;
        b->edges[lastEi].activeSlot = slot;
    }
    b->edges[ei].activeSlot = -1;
}
static inline void adjRemoveEdge(Body *b, int ni, uint16_t ei) {
    uint8_t c = b->adjCount[ni];
    for (uint8_t k=0; k<c; ++k) if (b->adjEdge[ni][k] == (uint16_t)ei) {
        uint8_t last = (uint8_t)(c-1);
        b->adjEdge[ni][k] = b->adjEdge[ni][last];
        b->adjNode[ni][k] = b->adjNode[ni][last];
        b->adjCount[ni] = last;
        return;
    }
}
static int nodeHashFindSlot(const Body *b, uint32_t key) {
    uint32_t h = nodeHash0(key);
    for (int i=0; i<NODEHASH_CAP; ++i) {
        uint32_t k = b->nodeHashKey[h];
        if (k == NODEKEY_EMPTY) return -1;
        if (k == key) return (int) h;
        h = (h + 1) & NODEHASH_MASK;
    }
    return -1;
}
static int nodeHashFindInsertSlot(const Body *b, uint32_t key) {
    uint32_t h = nodeHash0(key);
    int firstTomb = -1;
    for (int i=0; i<NODEHASH_CAP; ++i) {
        uint32_t k = b->nodeHashKey[h];
        if (k == NODEKEY_EMPTY) return (firstTomb >= 0)? firstTomb : (int)h;
        if (k == NODEKEY_TOMB && firstTomb < 0) firstTomb = (int)h;
        if (k == key) return (int)h;
        h = (h + 1) & NODEHASH_MASK;
    }
    return firstTomb;
}
static inline int nodeHashGet(const Body *b, Vec3Int coord) {
    uint32_t key = packCoord(coord);
    int hs = nodeHashFindSlot(b, key);
    if (hs < 0) return -1;
    return (int)b->nodeHashVal[hs];
}
static inline int nodeHashPut(Body *b, Vec3Int coord, int nodeIndex) {
    uint32_t key = packCoord(coord);
    int hs = nodeHashFindInsertSlot(b, key);
    if (hs < 0) return 0;
    if (b->nodeHashKey[hs] == key) return 0;
    b->nodeHashKey[hs] = key;
    b->nodeHashVal[hs] = (uint32_t)nodeIndex;
    return 1;
}
static int faceHashFindSlot(const Body *b, uint32_t key) {
    uint32_t h = faceHash0(key);
    for (int i=0; i<FACEHASH_CAP; ++i) {
        uint32_t k = b->faceHashKey[h];
        if (k == FACEKEY_EMPTY) return -1;
        if (k == key) return (int) h;
        h = (h + 1) & FACEHASH_MASK;
    }
    return -1;
}
static int faceHashFindInsertSlot(const Body *b, uint32_t key) {
    uint32_t h = faceHash0(key);
    int firstTomb = -1;
    for (int i=0; i<FACEHASH_CAP; ++i) {
        uint32_t k = b->faceHashKey[h];
        if (k == FACEKEY_EMPTY) return (firstTomb >= 0)? firstTomb : (int)h;
        if (k == FACEKEY_TOMB && firstTomb < 0) firstTomb = (int)h;
        if (k == key) return (int)h;
        h = (h + 1) & FACEHASH_MASK;
    }
    return firstTomb;
}
static inline void adjAdd(Body *b, int a, int c, uint16_t ei) {
    uint8_t ca = b->adjCount[a], cc = b->adjCount[c];
    if (ca >= ADJ_MAX || cc >= ADJ_MAX) return;
    b->adjNode[a][ca] = (uint16_t)c; b->adjEdge[a][ca] = ei; b->adjCount[a] = ca+1;
    b->adjNode[c][cc] = (uint16_t)a; b->adjEdge[c][cc] = ei; b->adjCount[c] = cc+1;
}
static inline void setEdgeType(Body *b, Edge *e, EdgeType t) {
    if (e->type == t) return;
    if (e->type == EDGE_BONE) {b->boneDeg[e->a]--; b->boneDeg[e->b]--;}
    switch (t) {
        case EDGE_NONE: {
            b->topoDirty = 1; 
            e->k = 0.0f;
            e->damping = 0.0f;
            e->hp = 0;
            uint16_t ei = (uint16_t)(e - b->edges);
            adjRemoveEdge(b, e->a, ei); adjRemoveEdge(b, e->b, ei);
            uint32_t key = edgeKey(e->a, e->b);
            int hs = edgeHashFindSlot(b, key);
            if (hs >= 0) {
                b->edgeHashKey[hs] = EDGEKEY_TOMB;
                b->edgeHashVal[hs] = 0;
            }
            activeRemoveEdge(b, (uint16_t)(e - b->edges));
        } break;
        case EDGE_TENDON: {
            e->k = 10.0f; 
            e->damping = 0.05f; 
            e->hp = 1; 
        } break;
        case EDGE_MUSCLE: {
            e->k = 4.0f; 
            e->damping = 0.03f; 
            e->hp = 1;
        } break;
        case EDGE_BONE:  {
            e->k = 0.0f; 
            e->damping = 0.0f;
            e->hp = 3;
            b->boneDeg[e->a]++; b->boneDeg[e->b]++;
        } break;
    }
    wakeBody(b);
    e->type = t;
}
static inline void setNodeType(Body *b, Node *n, NodeType t) {
    if (t==NODE_NORMAL) {
        n->radius = 0.12f;
        n->invMass = 1.0f;
    } else if (t==NODE_HEART) {
        n->radius = 0.16f;
        n->invMass = 0.5f;
    } else if (t==NODE_WEAPON) {
        n->radius = 0.20f;
        n->invMass = 1.25f;
    }
    n->type = t;
}
static inline void nodeAlive(Body *b, Node *n, int8_t is) {
    if (is) {
        n->alive = 1;
    } else {
        n->alive = 0;
        n->invMass = 0.0f;
        n->vel = v3(0, 0, 0);
        int ni = (int)(n - b->nodes);
        while (b->adjCount[ni] > 0) {
            uint16_t ei = b->adjEdge[ni][b->adjCount[ni] - 1];
            setEdgeType(b, &b->edges[ei], EDGE_NONE);
        }
        b->topoDirty = 1;
    }
}
static inline void jointCandRemove(Body *b, uint16_t n){
    int16_t slot = b->jointCandSlot[n];
    if (slot < 0) return;
    uint16_t last = b->jointCandIdx[--b->jointCandCount];
    if ((uint16_t)slot != b->jointCandCount){
        b->jointCandIdx[slot] = last;
        b->jointCandSlot[last] = slot;
    }
    b->jointCandSlot[n] = -1;
}
static inline void jointCandAdd(Body *b, uint16_t n){
    if (b->jointCandSlot[n] >= 0) return;
    b->jointCandSlot[n] = (int16_t)b->jointCandCount;
    b->jointCandIdx[b->jointCandCount++] = n;
}

static void computeBoneDegree(Body *b) {
    memset(b->boneDeg, 0, (size_t)b->nodeCount * sizeof(b->boneDeg[0]));
    for (uint16_t i = 0; i < b->activeEdgeCount; ++i) {
        Edge *e = &b->edges[b->activeEdgeIdx[i]];
        if (e->type != EDGE_BONE) continue;
        b->boneDeg[e->a]++; b->boneDeg[e->b]++;
    }
}
static void solveDistance(Node *a, Node *b, float rest, float stiffness) {
    Vec3 d = v3Sub(b->pos, a->pos);
    float dist = v3Len(d) + 1e-6f;
    float w1 = a->invMass, w2 = b->invMass, wsum = w1 + w2;
    if (wsum <= 0.0f) return;

    float C = dist - rest;                 // constraint error
    float s = (C / dist) * stiffness;      // how hard to correct

    // Move positions (weighted by inverse mass)
    a->pos = v3Add(a->pos, v3Scale(d,  ( w1 / wsum) * s));
    b->pos = v3Add(b->pos, v3Scale(d, -( w2 / wsum) * s));
}
static void solveBones(Body *b) {
    const int iters = 10;
    for (int it = 0; it < iters; ++it) {
        for (uint16_t i = 0; i < b->activeEdgeCount; ++i) {
            Edge *e = &b->edges[b->activeEdgeIdx[i]];
            if (e->type == EDGE_BONE) solveDistance(&b->nodes[e->a], &b->nodes[e->b], e->restLen, 1.0f);
        }
    }
}

static void recomputeHeartDist(Body *b) {
    uint16_t q[NODES_PER_BODY]; int qh=0, qt=0;
    for (int i=0; i<b->nodeCount;i++) b->nodes[i].distHeart=0;
    if (!b->nodes[0].alive || b->nodes[0].type!=NODE_HEART) return;
    b->nodes[0].distHeart=1; q[qt++]=0;

    while (qh < qt) {
        int u = q[qh++]; uint8_t du = b->nodes[u].distHeart;
        for (uint8_t k=0; k<b->adjCount[u]; ++k) {
            int v = b->adjNode[u][k];
            if (!b->nodes[v].alive) continue;
            if (b->nodes[v].distHeart == 0) {
                b->nodes[v].distHeart = du + 1; 
                q[qt++] = (uint16_t)v;
            }
        }
    }
}
static void cullDeadNodes(Body *b){
    for (int i=0;i<b->nodeCount;i++){
        Node *n = &b->nodes[i];
        if (!n->alive) continue;
        if (n->distHeart == 0) nodeAlive(b, n, 0);
    }
}

static inline Vec3Int v3IntMin4(Vec3Int a, Vec3Int b, Vec3Int c, Vec3Int d) {
    return v3Int(
        (int)fminf(a.x, fminf(b.x, fminf(c.x, d.x))),
        (int)fminf(a.y, fminf(b.y, fminf(c.y, d.y))),
        (int)fminf(a.z, fminf(b.z, fminf(c.z, d.z)))
    );
}
static inline Vec3Int v3IntMax4(Vec3Int a, Vec3Int b, Vec3Int c, Vec3Int d) {
    return v3Int(
        (int)fmaxf(a.x, fmaxf(b.x, fmaxf(c.x, d.x))),
        (int)fmaxf(a.y, fmaxf(b.y, fmaxf(c.y, d.y))),
        (int)fmaxf(a.z, fmaxf(b.z, fmaxf(c.z, d.z)))
    );
}
static inline int coordInUnitCube(Vec3Int p, Vec3Int o) {
    return (p.x >= o.x && p.x <= o.x+1 &&
            p.y >= o.y && p.y <= o.y+1 &&
            p.z >= o.z && p.z <= o.z+1);
}
static inline int cornerIndex(Vec3Int p, Vec3Int o) {
    int dx = p.x - o.x;
    int dy = p.y - o.y;
    int dz = p.z - o.z;
    return (dx<<2) | (dy<<1) | dz;
}
static inline Vec3Int cornerCoord(Vec3Int o, int idx) {
    return v3Int(
        o.x + ((idx>>2)&1),
        o.y + ((idx>>1)&1),
        o.z + (idx&1)
    );
}
static void buildCubeTets(int outTets[6][4], int parity) {
    if (parity == 0) {
        int T[6][4] = {
            {0,4,6,7},
            {0,6,2,7},
            {0,2,3,7},
            {0,3,1,7},
            {0,1,5,7},
            {0,5,4,7},
        };
        memcpy(outTets, T, sizeof(T));
    } else {
        int T[6][4] = {
            {4,0,2,3},
            {4,2,6,3},
            {4,6,7,3},
            {4,7,5,3},
            {4,5,1,3},
            {4,1,0,3},
        };
        memcpy(outTets, T, sizeof(T));
    }
}

static inline float latticeRestLen(const Node *a, const Node *b){
    int d2 = v3IntDist2(b->lattice, a->lattice);
    return REST_LEN * sqrtf((float)d2);
}
static int findMissingApexCoord(const Body *b, int x, int y, int z, Vec3Int *out) {
    const Node *A=&b->nodes[x], *B=&b->nodes[y], *C=&b->nodes[z];

    int minx = fmin(A->lattice.x, fmin(B->lattice.x, C->lattice.x)) - 2;
    int maxx = fmax(A->lattice.x, fmax(B->lattice.x, C->lattice.x)) + 2;
    int miny = fmin(A->lattice.y, fmin(B->lattice.y, C->lattice.y)) - 2;
    int maxy = fmax(A->lattice.y, fmax(B->lattice.y, C->lattice.y)) + 2;
    int minz = fmin(A->lattice.z, fmin(B->lattice.z, C->lattice.z)) - 2;
    int maxz = fmax(A->lattice.z, fmax(B->lattice.z, C->lattice.z)) + 2;

    for (int ix = minx; ix<=maxx; ++ix) {
        for (int iy = miny; iy<=maxy; ++iy) {
            for (int iz = minz; iz<=maxz; ++iz) {
                if ((ix + iy + iz) &1) continue;
                Vec3Int coord = v3Int(ix, iy, iz);
                if (v3IntDist2(coord, A->lattice) != 2) continue;
                if (v3IntDist2(coord, B->lattice) != 2) continue;
                if (v3IntDist2(coord, C->lattice) != 2) continue;
                if (nodeHashGet(b, coord) >= 0) continue;
                *out=coord; return 1;
            }
        }
    }
    return 0;
}
static void toggleFrontierFace(Body *b, uint16_t x, uint16_t y, uint16_t z, uint16_t inside) {
    if (x==y || y==z || z==x) return;
    uint32_t key = faceKey(x,y,z);
    int hs = faceHashFindSlot(b, key);

    if (hs >= 0) {
        uint16_t fi = b->faceHashVal[hs];
        if (fi < b->faceCount) b->frontier[fi].alive = 0;
        b->faceHashKey[hs] = FACEKEY_TOMB;
        b->faceHashVal[hs] = 0;
        return;
    }
    if (b->faceCount >= MAX_FACES) return;

    int ins = faceHashFindInsertSlot(b, key);
    if (ins < 0) return;

    uint16_t fi = (uint16_t)b->faceCount++;
    Face *f = &b->frontier[fi];
    f->a = x; f->b = y; f->c = z;
    f->inside = inside;
    f->alive = 1;

    b->faceHashKey[ins] = key;
    b->faceHashVal[ins] = fi;
}
static int findOutsideApexCoord_CubicTets(const Body *b, int ia, int ib, int ic, int iInside, Vec3Int *outApex) {
    Vec3Int A = b->nodes[ia].lattice;
    Vec3Int B = b->nodes[ib].lattice;
    Vec3Int C = b->nodes[ic].lattice;
    Vec3Int D = b->nodes[iInside].lattice;

    int minx = (int)fminf(A.x, fminf(B.x, C.x));
    int miny = (int)fminf(A.y, fminf(B.y, C.y));
    int minz = (int)fminf(A.z, fminf(B.z, C.z));
    int maxx = (int)fmaxf(A.x, fmaxf(B.x, C.x));
    int maxy = (int)fmaxf(A.y, fmaxf(B.y, C.y));
    int maxz = (int)fmaxf(A.z, fmaxf(B.z, C.z));

    if ((maxx - minx) > 1 || (maxy - miny) > 1 || (maxz - minz) > 1) return 0;

    Vec3Int candidates[16];
    int candCount = 0;

    for (int ox = minx - 1; ox <= minx; ++ox) {
        for (int oy = miny - 1; oy <= miny; ++oy) {
            for (int oz = minz - 1; oz <= minz; ++oz) {

                Vec3Int o = v3Int(ox, oy, oz);

                if (!coordInUnitCube(A, o) || !coordInUnitCube(B, o) || !coordInUnitCube(C, o)) continue;

                int a  = cornerIndex(A, o);
                int b2 = cornerIndex(B, o);
                int c2 = cornerIndex(C, o);

                int tets[6][4];
                buildCubeTets(tets, 0);

                for (int ti = 0; ti < 6; ++ti) {
                    int hasA=0, hasB=0, hasC=0;
                    int apexIdx = -1;
                    for (int k=0; k<4; ++k) {
                        int v = tets[ti][k];
                        if (v == a) hasA = 1;
                        else if (v == b2) hasB = 1;
                        else if (v == c2) hasC = 1;
                        else apexIdx = v;
                    }
                    if (!(hasA && hasB && hasC)) continue;
                    if (apexIdx < 0) continue;

                    Vec3Int apex = cornerCoord(o, apexIdx);

                    // Exclude the "inside" tetra apex, if this cube happens to be that one
                    if (apex.x == D.x && apex.y == D.y && apex.z == D.z) continue;

                    // If apex already exists in this body, face is not frontier
                    if (nodeHashGet(b, apex) >= 0) continue;
                    candidates[candCount++] = apex;
                }
            }
        }
    }
    if (candCount == 0) return 0;
    *outApex = candidates[rand() % candCount];
    return 1;
}
static EdgeType pickEdgeType(const Body *b, int a, int c) {
    float pB = b->genes.boneP, pT = b->genes.tendonP, pM = b->genes.muscleP; //edge type base probabilities
    //int ba = b->boneDeg[a], bc = b->boneDeg[c]; int be = ba + bc; //neighboring bone edge counts
    //float XBoneP = b->genes.skeletal * -be; //extra bone probability based on neighboirng bone edges

    float r = frand01();
    if (r < pB) return EDGE_BONE;
    if (r < pB + pT) return EDGE_TENDON;
    return EDGE_MUSCLE;
}
static int  addEdge(Body *b, int a, int c, EdgeType type) {
    if (a==c) return 0;
    if (a >= b->nodeCount || c >= b->nodeCount) return 0;
    if (b->edgeCount >= EDGES_PER_BODY) return 0;
    if (b->adjCount[a] >= ADJ_MAX) return 0; if (b->adjCount[c] >= ADJ_MAX) return 0;
    //Checks for edge preexistence and finds valid slot in hash
    uint32_t key = edgeKey(a, c);
    int h = edgeHashFindInsertSlot(b, key);
    if (h < 0) return 0;
    if (b->edgeHashKey[h] == key) return 0;

    //Prevents weapon nodes from gaining more than two connections
    if (b->nodes[a].type == NODE_WEAPON && b->adjCount[a] >= 2) return 0;
    if (b->nodes[c].type == NODE_WEAPON && b->adjCount[c] >= 2) return 0;

    int ei = b->edgeCount;
    Edge *e = &b->edges[b->edgeCount++];
    e->a=a; e->b=c;
    
    b->edgeHashKey[h] = key;
    b->edgeHashVal[h] = (uint16_t)ei;

    e->type = EDGE_NONE;
    e->activeSlot = b->activeEdgeCount;
    b->activeEdgeIdx[b->activeEdgeCount++] = (uint16_t)ei;
    setEdgeType(b, e, type);
    e->restLen = latticeRestLen(&b->nodes[a], &b->nodes[c]);
    e->muscleTargetMul=0.7f; e->muscleSignal=0.0f;

    adjAdd(b, a, c, (uint16_t)ei);

    return 1;
}
static void randomLinks(Body *b, int extraEdges) {
    for (int t=0;t<extraEdges;t++) {
        int a = rand() % b->nodeCount, c = rand() % b->nodeCount;
        if (!b->nodes[a].alive) continue;
        int best = -1; float bestScore = -1e30f;
        for (int tries = 0; tries < 8; ++tries) {
            int c = rand() % b->nodeCount;
            if (c == a || !b->nodes[c].alive) continue;
            float d = v3Dist(b->nodes[a].pos, b->nodes[c].pos);
            float score = -0.5f * d;
            if (score > bestScore) {bestScore = score; best = c;}
        }
        if (best >= 0) addEdge(b, a, c, pickEdgeType(b, a, c));
    }
}
static inline float parentScore(Body *b, int childIdx, int parentIdx) {
    const Genetics *g = &b->genes;
    const Node *child = &b->nodes[childIdx];
    const Node *parent = &b->nodes[parentIdx];
    if (!parent->alive) return -1e30f;

    float linearity = 2 + g->linearityBias * -(float)abs(b->adjCount[parentIdx] - g->linearity);

    return linearity;
}
static uint16_t seedNode(Body *b, Vec3Int lattice, NodeType t) {
    if (b->nodeCount >= NODES_PER_BODY) return UINT16_MAX;
    if (nodeHashGet(b, lattice) > 0) return UINT16_MAX;

    uint16_t i = b->nodeCount++;
    Node *n = &b->nodes[i];
    memset(n, 0, sizeof(*n));
    n->alive  = 1;
    n->owner = (uint16_t)b->id;
    n->lattice = lattice;
    n->pos = v3Add(b->nodes[0].pos, v3Scale(v3IntTov3(lattice), REST_LEN));
    n->vel = v3(0,0,0); 
    n->force = v3(0,0,0);
    n->distHeart = 0;
    
    setNodeType(b, &b->nodes[i], t);
    nodeHashPut(b, n->lattice, i);

    return i;
}
static int seedNewBoneonNode(Body *b, uint16_t J) {
    if (!b->nodes[J].alive) return 0;
    if (b->nodeCount + 3 > NODES_PER_BODY) return 0;
    if (b->adjCount[J] > 3) return 0;

    uint16_t A = b->adjNode[J][0], B = b->adjNode[J][1], C = b->adjNode[J][2];
    Vec3Int Jc = b->nodes[J].lattice;

    Vec3Int Ap = v3IntSub(v3IntScale(Jc,2), b->nodes[A].lattice);
    Vec3Int Bp = v3IntSub(v3IntScale(Jc,2), b->nodes[B].lattice);
    Vec3Int Cp = v3IntSub(v3IntScale(Jc,2), b->nodes[C].lattice);

    uint16_t iAp = seedNode(b, Ap, NODE_NORMAL);
    uint16_t iBp = seedNode(b, Bp, NODE_NORMAL);
    uint16_t iCp = seedNode(b, Cp, NODE_NORMAL);
    if (iAp==UINT16_MAX || iBp==UINT16_MAX || iCp==UINT16_MAX) return 0;

    addEdge(b, J,   iAp, EDGE_BONE);
    addEdge(b, J,   iBp, EDGE_BONE);
    addEdge(b, J,   iCp, EDGE_BONE);
    addEdge(b, iAp, iBp, EDGE_BONE);
    addEdge(b, iBp, iCp, EDGE_BONE);
    addEdge(b, iCp, iAp, EDGE_BONE);

    toggleFrontierFace(b, iAp, iBp, iCp, J);

    return 1;
}
static int seedNodeonFace(Body *b, int faceIndex) {
    Face *f = &b->frontier[faceIndex];
    if (!f->alive) return 0;
    if (b->nodeCount >= NODES_PER_BODY) return 0;
    Vec3Int apex;
    if (!findOutsideApexCoord_CubicTets(b, f->a, f->b, f->c, f->inside, &apex)){
        f->alive = 0;
        b->topoDirty = 1;
        return 0;
    }
    toggleFrontierFace(b, f->a, f->b, f->c, f->inside);

    uint16_t ni = seedNode(b, apex, NODE_NORMAL);

    addEdge(b, ni, f->a, EDGE_BONE /*pickEdgeType(b, ni, f->a)*/);
    addEdge(b, ni, f->b, EDGE_BONE /*pickEdgeType(b, ni, f->b)*/);
    addEdge(b, ni, f->c, EDGE_BONE /*pickEdgeType(b, ni, f->c)*/);

    f->alive = 0;
    toggleFrontierFace(b, f->a, f->b, ni, f->c);
    toggleFrontierFace(b, f->b, f->c, ni, f->a);
    toggleFrontierFace(b, f->c, f->a, ni, f->b);

    jointCandRemove(b, f->a);
    jointCandRemove(b, f->b);
    jointCandRemove(b, f->c);
    jointCandAdd(b, ni);

    b->topoDirty = 1;
    wakeBody(b);
    return 1;
}
static void skeletonLattice(Body *b, Vec3 center) {
    b->edgeCount = 0;
    b->activeEdgeCount = 0;
    b->faceCount = 0;
    b->jointCandCount = 0;
    b->nodeCount = 0;

    
    seedNode(b, v3Int(0,0,0), NODE_HEART);
    b->nodes[0].pos = center;
    seedNode(b, v3Int(1,0,0), NODE_NORMAL);
    seedNode(b, v3Int(1,1,0), NODE_NORMAL);
    seedNode(b, v3Int(1,1,1), NODE_NORMAL);

    jointCandAdd(b, 0); jointCandAdd(b, 1); jointCandAdd(b, 2); jointCandAdd(b, 3);

    addEdge(b, 0,1, EDGE_BONE); addEdge(b, 0,2, EDGE_BONE); addEdge(b, 0,3, EDGE_BONE); addEdge(b, 1,2, EDGE_BONE); addEdge(b, 1,3, EDGE_BONE); addEdge(b, 2,3, EDGE_BONE);
    toggleFrontierFace(b, 1,2,3, 0); toggleFrontierFace(b, 0,2,3, 1); toggleFrontierFace(b, 0,1,3, 2); toggleFrontierFace(b, 0,1,2, 3);

    if (b->faceCount == 0) {
        computeBoneDegree(b);
        return;
    }

    while (b->nodeCount < NODES_PER_BODY) {
            if (rand() % 6 > 0 && b->jointCandCount > 0) {
            int tries = 0;
            int fi = -1;
            while (tries++ < 64) {
                int r = rand() % b->faceCount;
                if (b->frontier[r].alive) {fi = r; break;}
            }
            if (fi < 0) break;
            seedNodeonFace(b, fi);
        } else {seedNewBoneonNode(b, b->jointCandIdx[rand() % b->jointCandCount]);}
    }
    computeBoneDegree(b);
}
static void generateDNA(Genetics *g) {
    g->linearity = irandRange(1, ADJ_MAX);
    g->linearityBias = frand01();
    g->skeletal = frand01();
    float cut1 = frand01(), cut2 = frand01();
    if (cut1 > cut2) {float temp=cut1; cut1=cut2; cut2=temp;}
    g->boneP = cut1; g->tendonP = cut2-cut1; g->muscleP = 1-cut2;
}
static void makeRandomOrganism(int slot, Vec3 center) {
    Body *b = &gBodies[slot];
    Genetics *g = &b->genes;
    b->id = (uint16_t)(slot + 1);
    b->edgeCount = 0; b->activeEdgeCount = 0;
    for (int i=0; i<NODES_PER_BODY; ++i) b->adjCount[i] = 0;
    for (int i=0; i<EDGEHASH_CAP; ++i) {
        b->edgeHashKey[i] = EDGEKEY_EMPTY;
        b->edgeHashVal[i] = 0;
    }
    for (int i=0; i<NODEHASH_CAP; ++i) {
        b->nodeHashKey[i] = NODEKEY_EMPTY;
        b->nodeHashVal[i] = 0;
    }
    for (int i=0; i<FACEHASH_CAP; ++i) {
        b->faceHashKey[i] = FACEKEY_EMPTY;
        b->faceHashVal[i] = 0;
    }
    for (int i=0; i<NODES_PER_BODY; ++i) b->jointCandSlot[i] = -1;
    b->sleeping = 0;
    b->sleepFrames = 0;
    b->topoDirty = 1;

    generateDNA(g);

    skeletonLattice(b, center);
    //randomLinks(b, 8 + (rand()%6));             // 8–13 extra edges

    bodyBounds(b);
}
static void populateWorld(int count) {
    if (count > MAX_BODIES) count = MAX_BODIES;
    gBodyCount = count;

    for (int i=0;i<count;i++){
        Vec3 c = v3((frand01()*2-1) * (WORLD_X*0.7f),
                    (frand01()*2-1) * (WORLD_Y*0.7f),
                    (frand01()*2-1) * (WORLD_Z*0.7f));
        makeRandomOrganism(i, c);
    }
}



static void applyImpulse(Node *n, Vec3 J){
    if (n->invMass <= 0.0f) return;
    n->vel = v3Add(n->vel, v3Scale(J, n->invMass));
}

static Vec3 closestPointSeg(Vec3 p, Vec3 a, Vec3 b, float *outT){
    Vec3 ab = v3Sub(b,a);
    float t = CLAMP01(v3Dot(v3Sub(p,a), ab) / (v3DotSq(ab) + 1e-8f));
    if (outT) *outT = t;
    return v3Add(a, v3Scale(ab, t));
}
static int  weaponHitsSeg(Vec3 d, float r2, Vec3 *outNor){
    float d2 = v3DotSq(d); if (d2 >= r2) return 0;
    float eps2 = r2 * 1e-8f; if (d2 < eps2) d2 = eps2;
    float invLen = invSqrt(d2, 1);
    *outNor = v3Scale(d, invLen); // Just normalize but using as much precomputed as possible
    return 1;
}
static void bounceWeaponVsEdge(Node *w, Node *a, Node *b, Vec3 n, float e){
    Vec3 vEdge = v3Scale(v3Add(a->vel, b->vel), 0.5f);
    Vec3 vRel  = v3Sub(w->vel, vEdge);
    float vn = v3Dot(vRel, n);
    if (vn >= 0.0f) return; // separating already

    float wsum = w->invMass + 0.5f*(a->invMass + b->invMass) + 1e-8f;
    float j = -(1.0f + e) * vn / wsum;
    Vec3 J = v3Scale(n, j);

    applyImpulse(w, J);
    applyImpulse(a, v3Scale(J, -0.5f));
    applyImpulse(b, v3Scale(J, -0.5f));
}
static void weaponStepAll(Body *self, int wi){
    Node *w = &self->nodes[wi];
    if (!w->alive || w->type != NODE_WEAPON) return;
    Vec3 p = w->pos;
    Vec3 r3 = v3(w->radius, w->radius, w->radius);

    for (int i=0; i<gBodyCount; ++i) {
        Body *other = &gBodies[i];
        if (self->sleeping && other->sleeping) continue;
        if (other->id == self->id) continue;
        if (!v3Tol(v3Sub(w->pos, other->bsCenter), w->radius + other->bsRadius + 0.25f)) continue;

        if (!v3FullMax(p, v3Sub(other->bbMin, r3)) || !v3FullMax(v3Add(other->bbMax, r3), p)) continue;

        for (int ei=0; ei<other->activeEdgeCount; ++ei){
            Edge *e = &other->edges[other->activeEdgeIdx[ei]];
            if (e->type == EDGE_NONE) continue;

            Node *a = &other->nodes[e->a], *b = &other->nodes[e->b];
            Vec3 hitP = closestPointSeg(w->pos, a->pos, b->pos, NULL);
            Vec3 nor; if (!weaponHitsSeg(v3Sub(w->pos, hitP), sqr(w->radius), &nor)) continue;

            e->hp -= 1;
            if (e->hp <= 0) {
                HitFX fx = FX_ON_BREAK[e->type];
                setEdgeType(other, e, EDGE_NONE);
                spawnBreakFX(hitP, nor, fx.count, fx.r, fx.g, fx.b, fx.a);
            } else { bounceWeaponVsEdge(w, a, b, nor, FX_ON_BREAK[e->type].bounceE);}
            return;
        }
    }
}

static void clearForces(Node* n, int count) {
    for (int i = 0; i < count; ++i) v3Zero(&n[i].force);
}
static void updateVelFromPos(Node *n, Vec3 *oldPos, int count, float dt) {
    float invDt = (dt > 0.0f) ? (1.0f / dt) : 0.0f;
    for (int i = 0; i < count; ++i) {
        n[i].vel = v3Scale(v3Sub(n[i].pos, oldPos[i]), invDt);
    }
}

static void applySpring(Node* a, Node* b, float rest, float k, float damp) {
    Vec3 pos = v3Sub(b->pos, a->pos);
    float dist = v3Len(pos) + 1e-6f;
    k *= dist / rest;
    pos = v3Scale(pos, 1/dist);
    Vec3 relVel = v3Sub(b->vel, a->vel);
    float vrel = v3Dot(pos, relVel);
    float f = k * (dist - rest) + damp * vrel;
    pos = v3Scale(pos, f);
    a->force = v3Add(a->force, pos);
    b->force = v3Sub(b->force, pos);
}
static void applyAllSprings(Body *b) {
    for (uint16_t i=0; i<b->activeEdgeCount; ++i) {
        Edge *e = &b->edges[b->activeEdgeIdx[i]];
        if (e->type == EDGE_BONE) continue;
        applySpring(&b->nodes[e->a], &b->nodes[e->b], e->restLen, e->k, e->damping);
    }
}
static void resolveBounds(Node *n) {
    const float bounce = 0.8f;

    if (n->pos.x < -WORLD_X) {
        n->pos.x = -WORLD_X;
        n->vel.x *= -bounce;
    } else if (n->pos.x > WORLD_X) {
        n->pos.x = WORLD_X;
        n->vel.x *= -bounce;
    }

    if (n->pos.y < -WORLD_Y) {
        n->pos.y = -WORLD_Y;
        n->vel.y *= -bounce;
    } else if (n->pos.y > WORLD_Y) {
        n->pos.y = WORLD_Y;
        n->vel.y *= -bounce;
    }

    if (n->pos.z < -WORLD_Z) {
        n->pos.z = -WORLD_Z;
        n->vel.z *= -bounce;
    } else if (n->pos.z > WORLD_Z) {
        n->pos.z = WORLD_Z;
        n->vel.z *= -bounce;
    }
}
static void integrate(Node* n, int count, float dt, const uint8_t *boneDeg) {
    float damp = fmaxf(0.0f, 1.0f - GLOBAL_DAMPING * dt);

    for (int i = 0; i < count; ++i) {
        if (!n[i].alive) continue;
        float invm = n[i].invMass / (1.0f + (float)boneDeg[i]);
        n[i].vel = v3Scale(v3Add(n[i].vel, v3Scale(n[i].force, invm * dt)), damp);
        n[i].pos = v3Add(n[i].pos, v3Scale(n[i].vel, dt));
        resolveBounds(&n[i]);
    }
}
static void physicsTickBody(Body *b, float dt) {
    if (b->sleeping) return;

    Vec3 oldPos[NODES_PER_BODY];
    for (int i = 0; i < b->nodeCount; ++i) oldPos[i] = b->nodes[i].pos;

    clearForces(b->nodes, b->nodeCount);
    applyAllSprings(b);
    integrate(b->nodes, b->nodeCount, dt, b->boneDeg);

    solveBones(b);
    updateVelFromPos(b->nodes, oldPos, b->nodeCount, dt);

    if (bodyIsQuiet(b)) {
        b->sleepFrames++;
        if (b->sleepFrames >= SLEEP_FRAMES) {
            b->sleeping = 1;
            for (int i=0;i<b->nodeCount;i++) v3Zero(&b->nodes[i].vel);
            return;
        }
    } else {
        b->sleepFrames = 0;
    }

    bodyBounds(b);
}



static void kickBody(Body *b, float impulse) {
    wakeBody(b);
    for (int i = 0; i < b->nodeCount; ++i) {
        if (b->nodes[i].invMass <= 0.0f) continue;
        b->nodes[i].vel.x += (frand01()*2.0f - 1.0f) * impulse;
        b->nodes[i].vel.y += (frand01()*2.0f - 1.0f) * impulse;
        b->nodes[i].vel.z += (frand01()*2.0f - 1.0f) * impulse;
    }
}
static void offsetBody(Body *b, float amount) {
    wakeBody(b);
    for (int i = 0; i < b->nodeCount; ++i) {
        if (b->nodes[i].invMass <= 0.0f) continue;
        b->nodes[i].pos.x += (frand01()*2.0f - 1.0f) * amount;
        b->nodes[i].pos.y += (frand01()*2.0f - 1.0f) * amount;
        b->nodes[i].pos.z += (frand01()*2.0f - 1.0f) * amount;
    }
}
static int handleInput (Camera *cam) {
    SDL_Event e; static int md=0;
    while (SDL_PollEvent(&e)) {
        if (e.type == SDL_QUIT) return 0;
        else if (e.type == SDL_MOUSEBUTTONDOWN && e.button.button == SDL_BUTTON_LEFT) {md = 1;}
        else if (e.type == SDL_MOUSEBUTTONUP && e.button.button == SDL_BUTTON_LEFT) {md = 0;}
        else if (e.type == SDL_MOUSEMOTION && md) {
            cam->yaw     -= e.motion.xrel * 0.01f;
            cam->pitch   -= e.motion.yrel * 0.01f;
        } else if (e.type == SDL_MOUSEWHEEL) {
            cam->focal = CLAMP(cam->focal * ((e.wheel.y>0)? 0.95f : (e.wheel.y<0)? 1.05f : 1.0f), 1.0f, 30.0f);
        }
        else if (e.type == SDL_KEYDOWN) {
            if (e.key.keysym.sym == SDLK_ESCAPE) return 0;
            if (e.key.keysym.sym == SDLK_k) for (int i=0;i<gBodyCount;i++) kickBody(&gBodies[i], 2.0f);
            if (e.key.keysym.sym == SDLK_o) for (int i=0;i<gBodyCount;i++) offsetBody(&gBodies[i], 0.25f);
        }
    }

    return 1;
}

int main(int argc, char **argv) {
    srand(time(NULL));
    (void)argc; (void)argv;

    SDL_Window *win = NULL;
    SDL_GLContext gl = NULL;
    if (!initSDL(&win, &gl)) return 1;
    initNodeUVConst();

    Camera cam;
    cam.center = v3(0,0,0);
    cam.scale  = 80.0f;
    cam.yaw    = 0.0f;
    cam.pitch  = 0.35f;
    cam.focal = 6.0f;
    cam.nearZ = 0.1f;
    setPerspective(win);

    populateWorld(12);

    Uint64 prev = SDL_GetPerformanceCounter();
    double freq = (double)SDL_GetPerformanceFrequency();

    int running = 1;
    while (running) {
        if (!handleInput(&cam)) running = 0;
        Uint64 now = SDL_GetPerformanceCounter();
        float dt = CLAMP((now - prev) / freq, 0.0f, 0.05f);
        prev = now;

        for (int i=0; i<gBodyCount; ++i) {
            physicsTickBody(&gBodies[i], dt);
        }

        for (int bi=0; bi<gBodyCount; ++bi) {
            Body *self = &gBodies[bi];
            for (int ni=0; ni<self->nodeCount; ++ni) {
                if (self->nodes[ni].type == NODE_WEAPON) weaponStepAll(self, ni);
            }
        }
        for (int i=0;i<gBodyCount; ++i) {
            Body *b = &gBodies[i];
            if (!b->topoDirty) continue;
            recomputeHeartDist(b);
            cullDeadNodes(b);
            b->topoDirty = 0;
        }

        updateParticles(dt);

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        cameraTranscendentals(&cam);
        buildCameraNodes(&cam);

        drawEdges(&cam);
        drawNodes(&cam);

        drawParticles(&cam);

        SDL_GL_SwapWindow(win);
    }

    shutdownSDL(win, gl);
    return 0;
}

/* TO DO
    force generation of tendons around joints; increase spring strength significantly, hopefully enough to prevent inverting
    Tag joints and check for collision solving every so often
    add bone formation vs bone growth probabilities to DNA and impliment
    reenable muscle generation
    add ambient sensory inputs such as proximity to nodes of interest to allow future implimentation of basic behaviors
    add food and hunger stimulation for a simple goal; possibly force seperated nodes to become bodyless food nodes instead of vanishing
    complete generateDNA and impliment genetics for greater diversity and an evolutionary and reproductive basis
    add a reproductive node to allow for the introduction of evolutionary behaviors
*/

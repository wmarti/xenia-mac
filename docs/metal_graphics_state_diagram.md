# Metal Graphics Backend State Machine Diagram

## Overview

This document describes the state machine architecture of Xenia's Metal graphics backend and its relationship to the shared GPU backend code in `src/xenia/gpu`.

---

## 1. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              XENIA EMULATOR                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                         GAME (Xbox 360)                               │   │
│  │                                                                       │   │
│  │   Writes PM4 commands to ring buffer → Updates write pointer register │   │
│  └───────────────────────────────┬──────────────────────────────────────┘   │
│                                  │                                           │
│                                  ▼                                           │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    SHARED GPU BACKEND (src/xenia/gpu)                 │   │
│  │  ┌────────────────────────────────────────────────────────────────┐  │   │
│  │  │  GraphicsSystem (Base)         RegisterFile                    │  │   │
│  │  │  - System initialization       - 20,483 GPU registers         │  │   │
│  │  │  - Register I/O                - State storage                 │  │   │
│  │  │  - Interrupt handling          - Fetch descriptors             │  │   │
│  │  │  - VBlank management           - Shader constants              │  │   │
│  │  └────────────────────────────────────────────────────────────────┘  │   │
│  │  ┌────────────────────────────────────────────────────────────────┐  │   │
│  │  │  CommandProcessor (Base)                                       │  │   │
│  │  │  - PM4 packet parsing                                          │  │   │
│  │  │  - Worker thread execution                                     │  │   │
│  │  │  - Ring buffer management                                      │  │   │
│  │  │  - Abstract: IssueDraw(), IssueCopy(), LoadShader()           │  │   │
│  │  └────────────────────────────────────────────────────────────────┘  │   │
│  └───────────────────────────────┬──────────────────────────────────────┘   │
│                                  │ inherits                                  │
│                                  ▼                                           │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                 METAL BACKEND (src/xenia/gpu/metal)                   │   │
│  │  ┌────────────────────────────────────────────────────────────────┐  │   │
│  │  │  MetalGraphicsSystem          MetalCommandProcessor            │  │   │
│  │  │  - Metal device setup         - Metal command buffer mgmt     │  │   │
│  │  │  - Command queue creation     - Render encoder management     │  │   │
│  │  │  - Presenter integration      - Pipeline state caching        │  │   │
│  │  └────────────────────────────────────────────────────────────────┘  │   │
│  │  ┌────────────────────────────────────────────────────────────────┐  │   │
│  │  │  MetalSharedMemory   MetalTextureCache   MetalRenderTargetCache│  │   │
│  │  │  MetalShader         MetalPrimitiveProcessor                   │  │   │
│  │  │  MetalShaderConverter (DXBC → DXIL → Metal IR)                │  │   │
│  │  └────────────────────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                  │                                           │
│                                  ▼                                           │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                          METAL API (macOS)                            │   │
│  │                                                                       │   │
│  │   MTL::Device → MTL::CommandQueue → MTL::CommandBuffer → GPU         │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Command Processor State Machine

### 2.1 Worker Thread Lifecycle

```
                              ┌─────────────────┐
                              │   INITIALIZED   │
                              │                 │
                              │ SetupContext()  │
                              └────────┬────────┘
                                       │
                                       ▼
                              ┌─────────────────┐
                              │     IDLE        │◄─────────────────────┐
                              │                 │                      │
                              │ Waiting for     │                      │
                              │ write pointer   │                      │
                              │ update          │                      │
                              └────────┬────────┘                      │
                                       │                               │
                         Write pointer updated                         │
                                       │                               │
                                       ▼                               │
                              ┌─────────────────┐                      │
                              │   EXECUTING     │                      │
                              │                 │                      │
                              │ ExecutePrimary  │                      │
                              │ Buffer()        │                      │
                              └────────┬────────┘                      │
                                       │                               │
                                       ▼                               │
                              ┌─────────────────┐                      │
                              │ PACKET_DISPATCH │──────────────────────┤
                              │                 │   All packets done   │
                              │ ExecutePacket() │                      │
                              └────────┬────────┘                      │
                                       │                               │
                        ┌──────────────┼──────────────┐                │
                        │              │              │                │
                        ▼              ▼              ▼                │
                   ┌─────────┐   ┌─────────┐   ┌─────────┐             │
                   │ Type 0  │   │ Type 1  │   │ Type 3  │             │
                   │ RegWr   │   │ RegWr+  │   │  PM4    │             │
                   └────┬────┘   └────┬────┘   └────┬────┘             │
                        │             │             │                  │
                        └──────────────┴─────────────┘                 │
                                       │                               │
                                       ▼                               │
                              ┌─────────────────┐                      │
                              │  UPDATE_STATE   │──────────────────────┘
                              │                 │
                              │ Update read ptr │
                              │ Check for more  │
                              └─────────────────┘
                                       │
                          Shutdown requested
                                       │
                                       ▼
                              ┌─────────────────┐
                              │   SHUTDOWN      │
                              │                 │
                              │ ShutdownContext │
                              └─────────────────┘
```

### 2.2 PM4 Packet Type 3 Command Dispatch

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         PM4 TYPE 3 COMMANDS                                 │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        DRAW COMMANDS                                 │   │
│  │                                                                      │   │
│  │  PM4_DRAW_INDX ──────┐                                              │   │
│  │  PM4_DRAW_INDX_2 ────┼───► ExecutePacketType3Draw()                 │   │
│  │  PM4_VIZ_QUERY ──────┘           │                                  │   │
│  │                                  ▼                                  │   │
│  │                         ┌────────────────┐                          │   │
│  │                         │  IssueDraw()   │ ◄── Backend Override     │   │
│  │                         │  (abstract)    │                          │   │
│  │                         └────────────────┘                          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        COPY/RESOLVE COMMANDS                         │   │
│  │                                                                      │   │
│  │  PM4_EVENT_WRITE_SHD ───► ExecutePacketType3_EVENT_WRITE_SHD()      │   │
│  │                                  │                                  │   │
│  │                                  ▼                                  │   │
│  │                         ┌────────────────┐                          │   │
│  │                         │  IssueCopy()   │ ◄── Backend Override     │   │
│  │                         │  (abstract)    │                          │   │
│  │                         └────────────────┘                          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        STATE COMMANDS                                │   │
│  │                                                                      │   │
│  │  PM4_SET_CONSTANT ────────► Write shader constants                  │   │
│  │  PM4_SET_CONSTANT2 ───────► Write shader constants (alt)            │   │
│  │  PM4_SET_SHADER_CONSTANTS ► Write shader constant arrays            │   │
│  │  PM4_LOAD_ALU_CONSTANT ───► Load ALU constants from memory          │   │
│  │  PM4_INVALIDATE_STATE ────► Cache flush                             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        SYNC COMMANDS                                 │   │
│  │                                                                      │   │
│  │  PM4_WAIT_REG_MEM ────────► GPU waits for condition                 │   │
│  │  PM4_INTERRUPT ───────────► Generate CPU interrupt                  │   │
│  │  PM4_MEM_WRITE ───────────► Write value to memory                   │   │
│  │  PM4_COND_WRITE ──────────► Conditional memory write                │   │
│  │  PM4_REG_TO_MEM ──────────► Copy register to memory                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        SHADER COMMANDS                               │   │
│  │                                                                      │   │
│  │  PM4_IM_LOAD ─────────────► Load shader microcode from memory       │   │
│  │  PM4_IM_LOAD_IMMEDIATE ───► Load shader microcode from packet       │   │
│  │                                  │                                  │   │
│  │                                  ▼                                  │   │
│  │                         ┌────────────────┐                          │   │
│  │                         │ LoadShader()   │ ◄── Backend Override     │   │
│  │                         │ (abstract)     │                          │   │
│  │                         └────────────────┘                          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        FLOW CONTROL                                  │   │
│  │                                                                      │   │
│  │  PM4_INDIRECT_BUFFER ─────► Call subroutine buffer                  │   │
│  │  PM4_INDIRECT_BUFFER_PFD ─► Prefetch indirect buffer                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Register File State

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         REGISTER FILE (20,483 registers)                    │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  SHADER CONSTANTS (0x4000 - 0x4FFF)                                  │   │
│  │                                                                      │   │
│  │  ├── Float Constants (512 x vec4)   SHADER_CONSTANT_000_X - 1FF_W   │   │
│  │  ├── Bool Constants                 SHADER_CONSTANT_BOOL            │   │
│  │  └── Loop Constants                 SHADER_CONSTANT_LOOP            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  FETCH CONSTANTS (0x4800 - 0x49FF)                                   │   │
│  │                                                                      │   │
│  │  ├── Vertex Fetch (96 slots)        GetVertexFetch(index)           │   │
│  │  └── Texture Fetch (32 slots)       GetTextureFetch(index)          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  RENDER STATE                                                        │   │
│  │                                                                      │   │
│  │  ├── RB_DEPTHCONTROL ────► Depth/stencil enable, compare funcs      │   │
│  │  ├── RB_BLENDCONTROL ────► Per-RT blend equations                   │   │
│  │  ├── RB_COLORCONTROL ────► Alpha test, pixel kill                   │   │
│  │  ├── RB_COLOR_MASK ──────► Write masks per RT                       │   │
│  │  ├── RB_MODECONTROL ─────► EDRAM mode, early-Z                      │   │
│  │  └── RB_SURFACE_INFO ────► Render target format, MSAA               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  PRIMITIVE ASSEMBLY (PA)                                             │   │
│  │                                                                      │   │
│  │  ├── PA_SU_SC_MODE_CNTL ─► Cull mode, polygon mode, provoking vtx   │   │
│  │  ├── PA_CL_CLIP_CNTL ────► Clip plane enables, user clip planes     │   │
│  │  ├── PA_CL_VTE_CNTL ─────► Viewport transform enable                │   │
│  │  └── PA_SC_WINDOW_* ─────► Scissor, viewport offset                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  VERTEX/GEOMETRY TESSELLATION (VGT)                                  │   │
│  │                                                                      │   │
│  │  ├── VGT_DRAW_INITIATOR ─► Primitive type, source select            │   │
│  │  ├── VGT_OUTPUT_PATH_CNTL ► Tessellation enable                     │   │
│  │  ├── VGT_HOS_* ──────────► Tessellation parameters                  │   │
│  │  └── VGT_PRIMITIVEID_EN ─► Primitive ID generation                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  SHADER PROGRAM                                                      │   │
│  │                                                                      │   │
│  │  ├── SQ_PROGRAM_CNTL ────► Num interpolators, param_gen pos         │   │
│  │  ├── SQ_CONTEXT_MISC ────► Pixel shader export mode                 │   │
│  │  └── SQ_INTERPOLATOR_* ──► Interpolation modes                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  DISPLAY CONTROLLER                                                  │   │
│  │                                                                      │   │
│  │  ├── DC_LUT_* ───────────► Gamma ramp tables (256-entry, PWL)       │   │
│  │  └── DC_OUTPUT_* ────────► Output configuration                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Metal Command Processor State Machine

### 4.1 Command Buffer Lifecycle

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    METAL COMMAND BUFFER STATE MACHINE                       │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│      ┌────────────────┐                                                     │
│      │    NO_BUFFER   │◄──────────────────────────────────────────┐        │
│      │                │                                            │        │
│      │ current_cmd_   │                                            │        │
│      │ buffer_ = null │                                            │        │
│      └───────┬────────┘                                            │        │
│              │                                                     │        │
│   EnsureCommandBuffer()                                            │        │
│              │                                                     │        │
│              ▼                                                     │        │
│      ┌────────────────┐                                            │        │
│      │ BUFFER_ACTIVE  │                                            │        │
│      │                │                                            │        │
│      │ Command buffer │                                            │        │
│      │ created, ready │                                            │        │
│      │ for encoding   │                                            │        │
│      └───────┬────────┘                                            │        │
│              │                                                     │        │
│    ┌─────────┴─────────┐                                           │        │
│    │                   │                                           │        │
│    ▼                   ▼                                           │        │
│ ┌──────────┐    ┌──────────────┐                                   │        │
│ │ ENCODING │    │  BLIT_PASS   │                                   │        │
│ │ RENDER   │    │              │                                   │        │
│ │ PASS     │    │ Blit encoder │                                   │        │
│ └────┬─────┘    │ for copies   │                                   │        │
│      │          └──────────────┘                                   │        │
│      │                                                             │        │
│   EndRenderEncoder()                                               │        │
│      │                                                             │        │
│      ▼                                                             │        │
│      ┌────────────────┐                                            │        │
│      │ BUFFER_READY   │                                            │        │
│      │                │                                            │        │
│      │ All encoders   │                                            │        │
│      │ ended          │                                            │        │
│      └───────┬────────┘                                            │        │
│              │                                                     │        │
│   EndCommandBuffer() / commit()                                    │        │
│              │                                                     │        │
│              ▼                                                     │        │
│      ┌────────────────┐                                            │        │
│      │   SUBMITTED    │                                            │        │
│      │                │                                            │        │
│      │ Waiting for    │                                            │        │
│      │ GPU execution  │                                            │        │
│      └───────┬────────┘                                            │        │
│              │                                                     │        │
│   addCompletedHandler() callback                                   │        │
│              │                                                     │        │
│              ▼                                                     │        │
│      ┌────────────────┐                                            │        │
│      │   COMPLETED    │────────────────────────────────────────────┘        │
│      │                │                                                     │
│      │ ProcessCompleted                                                     │
│      │ Submissions()  │                                                     │
│      └────────────────┘                                                     │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Render Encoder State

```
┌────────────────────────────────────────────────────────────────────────────┐
│                      RENDER ENCODER STATE MACHINE                           │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐                                                        │
│  │   NO_ENCODER    │◄────────────────────────────────────────────┐         │
│  │                 │                                              │         │
│  │ current_render_ │                                              │         │
│  │ encoder_ = null │                                              │         │
│  └────────┬────────┘                                              │         │
│           │                                                       │         │
│  GetCurrentRenderPassDescriptor()                                 │         │
│  + renderCommandEncoder()                                         │         │
│           │                                                       │         │
│           ▼                                                       │         │
│  ┌─────────────────┐                                              │         │
│  │ ENCODER_ACTIVE  │                                              │         │
│  │                 │                                              │         │
│  │ Ready for draw  │                                              │         │
│  │ commands        │                                              │         │
│  └────────┬────────┘                                              │         │
│           │                                                       │         │
│           ├────────────────────────────────────────┐              │         │
│           │                                        │              │         │
│           ▼                                        ▼              │         │
│  ┌─────────────────┐                      ┌─────────────────┐     │         │
│  │  STATE_BINDING  │                      │ RESOURCE_TRACK  │     │         │
│  │                 │                      │                 │     │         │
│  │ setRenderPipeline                      │ useResource()   │     │         │
│  │ setDepthStencil │                      │ useHeap()       │     │         │
│  │ setVertexBuffer │                      │                 │     │         │
│  │ setFragmentBuffer                      │ render_encoder_ │     │         │
│  │ setFragmentTexture                     │ _resource_usage_│     │         │
│  └────────┬────────┘                      └────────┬────────┘     │         │
│           │                                        │              │         │
│           └────────────────┬───────────────────────┘              │         │
│                            │                                      │         │
│                            ▼                                      │         │
│                   ┌─────────────────┐                             │         │
│                   │  DRAW_INDEXED   │                             │         │
│                   │                 │                             │         │
│                   │ drawIndexed     │                             │         │
│                   │ Primitives()    │                             │         │
│                   └────────┬────────┘                             │         │
│                            │                                      │         │
│                            ├──────────── More draws ──────────────┤         │
│                            │                                      │         │
│                   EndRenderEncoder()                              │         │
│                            │                                      │         │
│                            ▼                                      │         │
│                   ┌─────────────────┐                             │         │
│                   │ ENCODER_ENDED   │─────────────────────────────┘         │
│                   │                 │                                       │
│                   │ endEncoding()   │                                       │
│                   │ Reset tracking  │                                       │
│                   └─────────────────┘                                       │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Draw Call Flow (Shared → Metal)

```
┌────────────────────────────────────────────────────────────────────────────┐
│                           DRAW CALL FLOW                                    │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    SHARED (CommandProcessor)                         │   │
│  │                                                                      │   │
│  │   PM4_DRAW_INDX packet received                                      │   │
│  │           │                                                          │   │
│  │           ▼                                                          │   │
│  │   ExecutePacketType3Draw()                                           │   │
│  │           │                                                          │   │
│  │           ├── Extract primitive type from VGT_DRAW_INITIATOR        │   │
│  │           ├── Extract index count                                    │   │
│  │           ├── Setup IndexBufferInfo (format, endian, address)       │   │
│  │           │                                                          │   │
│  │           ▼                                                          │   │
│  │   IssueDraw(prim_type, index_count, index_buffer_info, major_mode)  │   │
│  │           │                                                          │   │
│  └───────────┼──────────────────────────────────────────────────────────┘   │
│              │ virtual call                                                 │
│              ▼                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                 METAL (MetalCommandProcessor::IssueDraw)             │   │
│  │                                                                      │   │
│  │   ┌─────────────────────────────────────────────────────────────┐   │   │
│  │   │ 1. SHADER LOADING                                            │   │   │
│  │   │                                                              │   │   │
│  │   │    LoadShader(vertex) ──► MetalShader                        │   │   │
│  │   │    LoadShader(pixel)  ──► MetalShader                        │   │   │
│  │   │                                                              │   │   │
│  │   │    ┌──────────────────────────────────────────────────┐     │   │   │
│  │   │    │  Shader Translation Pipeline:                     │     │   │   │
│  │   │    │                                                   │     │   │   │
│  │   │    │  Xbox 360 Microcode                               │     │   │   │
│  │   │    │        │                                          │     │   │   │
│  │   │    │        ▼ DxbcShaderTranslator                     │     │   │   │
│  │   │    │  DXBC Bytecode                                    │     │   │   │
│  │   │    │        │                                          │     │   │   │
│  │   │    │        ▼ DxbcToDxilConverter                      │     │   │   │
│  │   │    │  DXIL (DirectX IL)                                │     │   │   │
│  │   │    │        │                                          │     │   │   │
│  │   │    │        ▼ MetalShaderConverter (Apple MSC)         │     │   │   │
│  │   │    │  Metal IR Library                                 │     │   │   │
│  │   │    └──────────────────────────────────────────────────┘     │   │   │
│  │   └─────────────────────────────────────────────────────────────┘   │   │
│  │                                                                      │   │
│  │   ┌─────────────────────────────────────────────────────────────┐   │   │
│  │   │ 2. PRIMITIVE PROCESSING                                      │   │   │
│  │   │                                                              │   │   │
│  │   │    MetalPrimitiveProcessor::Process()                        │   │   │
│  │   │        │                                                     │   │   │
│  │   │        ├── Convert triangle fans → triangle lists           │   │   │
│  │   │        ├── Handle 32-bit indices via indirection            │   │   │
│  │   │        ├── Handle line loops (closing edge)                 │   │   │
│  │   │        └── Primitive restart handling                       │   │   │
│  │   └─────────────────────────────────────────────────────────────┘   │   │
│  │                                                                      │   │
│  │   ┌─────────────────────────────────────────────────────────────┐   │   │
│  │   │ 3. RENDER TARGET SETUP                                       │   │   │
│  │   │                                                              │   │   │
│  │   │    MetalRenderTargetCache::Update()                          │   │   │
│  │   │        │                                                     │   │   │
│  │   │        ├── Parse RB_SURFACE_INFO, RB_COLOR_INFO             │   │   │
│  │   │        ├── Resolve EDRAM aliasing                            │   │   │
│  │   │        ├── Create/retrieve MTL::Texture for each RT         │   │   │
│  │   │        └── Build MTL::RenderPassDescriptor                  │   │   │
│  │   └─────────────────────────────────────────────────────────────┘   │   │
│  │                                                                      │   │
│  │   ┌─────────────────────────────────────────────────────────────┐   │   │
│  │   │ 4. TEXTURE BINDING                                           │   │   │
│  │   │                                                              │   │   │
│  │   │    MetalTextureCache::RequestTextures()                      │   │   │
│  │   │        │                                                     │   │   │
│  │   │        ├── Parse texture fetch constants                     │   │   │
│  │   │        ├── Upload/convert guest textures                     │   │   │
│  │   │        └── Populate descriptor heap slots                    │   │   │
│  │   └─────────────────────────────────────────────────────────────┘   │   │
│  │                                                                      │   │
│  │   ┌─────────────────────────────────────────────────────────────┐   │   │
│  │   │ 5. PIPELINE STATE                                            │   │   │
│  │   │                                                              │   │   │
│  │   │    GetOrCreatePipelineState()                                │   │   │
│  │   │        │                                                     │   │   │
│  │   │        ├── Compute pipeline key from:                        │   │   │
│  │   │        │     - Vertex/pixel shader translations              │   │   │
│  │   │        │     - Render target formats                         │   │   │
│  │   │        │     - Blend state (RB_BLENDCONTROL)                 │   │   │
│  │   │        │     - Color mask                                    │   │   │
│  │   │        │     - MSAA sample count                             │   │   │
│  │   │        │                                                     │   │   │
│  │   │        ├── Check pipeline_cache_                             │   │   │
│  │   │        │                                                     │   │   │
│  │   │        └── If miss: Create MTL::RenderPipelineState         │   │   │
│  │   │                                                              │   │   │
│  │   │    ApplyDepthStencilState()                                  │   │   │
│  │   │        │                                                     │   │   │
│  │   │        └── Get/create MTL::DepthStencilState                │   │   │
│  │   └─────────────────────────────────────────────────────────────┘   │   │
│  │                                                                      │   │
│  │   ┌─────────────────────────────────────────────────────────────┐   │   │
│  │   │ 6. CONSTANT BUFFER SETUP                                     │   │   │
│  │   │                                                              │   │   │
│  │   │    UpdateSystemConstantValues()                              │   │   │
│  │   │        │                                                     │   │   │
│  │   │        ├── Populate system_constants_ struct                 │   │   │
│  │   │        ├── Viewport transform                                │   │   │
│  │   │        ├── Alpha test reference                              │   │   │
│  │   │        ├── Point sprite size                                 │   │   │
│  │   │        └── EDRAM parameters                                  │   │   │
│  │   │                                                              │   │   │
│  │   │    Copy float constants from RegisterFile → uniforms_buffer_ │   │   │
│  │   └─────────────────────────────────────────────────────────────┘   │   │
│  │                                                                      │   │
│  │   ┌─────────────────────────────────────────────────────────────┐   │   │
│  │   │ 7. ENCODE DRAW                                               │   │   │
│  │   │                                                              │   │   │
│  │   │    EnsureCommandBuffer()                                     │   │   │
│  │   │    renderCommandEncoder()                                    │   │   │
│  │   │        │                                                     │   │   │
│  │   │        ├── setRenderPipelineState(pipeline)                  │   │   │
│  │   │        ├── setDepthStencilState(ds_state)                    │   │   │
│  │   │        ├── setVertexBuffer(shared_memory_)                   │   │   │
│  │   │        ├── setFragmentBuffer(uniforms_buffer_)               │   │   │
│  │   │        ├── setFragmentTexture(...)                           │   │   │
│  │   │        │                                                     │   │   │
│  │   │        └── drawIndexedPrimitives(...)                        │   │   │
│  │   └─────────────────────────────────────────────────────────────┘   │   │
│  │                                                                      │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Metal Caching Hierarchy

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         METAL CACHING LAYERS                                │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        SHADER CACHE                                  │   │
│  │                                                                      │   │
│  │  shader_cache_: map<uint64_t (ucode_hash), MetalShader>             │   │
│  │                                                                      │   │
│  │  MetalShader contains:                                               │   │
│  │    ├── Original Xbox 360 microcode                                   │   │
│  │    ├── Translated DXBC bytecode                                      │   │
│  │    └── MetalTranslation(s) per modification:                         │   │
│  │          ├── DXIL bytecode                                           │   │
│  │          ├── MTL::Library (compiled Metal IR)                        │   │
│  │          └── Reflection info                                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                       PIPELINE CACHE                                 │   │
│  │                                                                      │   │
│  │  pipeline_cache_: map<uint64_t, MTL::RenderPipelineState*>          │   │
│  │                                                                      │   │
│  │  Key derived from:                                                   │   │
│  │    ├── Vertex shader translation pointer                             │   │
│  │    ├── Pixel shader translation pointer                              │   │
│  │    ├── Render target formats (color[4], depth, stencil)             │   │
│  │    ├── Blend state per RT                                            │   │
│  │    ├── Color write masks                                             │   │
│  │    └── MSAA sample count                                             │   │
│  │                                                                      │   │
│  │  geometry_pipeline_cache_: map<uint64_t, GeometryPipelineState>     │   │
│  │  tessellation_pipeline_cache_: map<uint64_t, TessellationPipeline>  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    DEPTH/STENCIL STATE CACHE                         │   │
│  │                                                                      │   │
│  │  depth_stencil_state_cache_: map<DepthStencilStateKey, DSState*>    │   │
│  │                                                                      │   │
│  │  Key includes:                                                       │   │
│  │    ├── RB_DEPTHCONTROL bits                                          │   │
│  │    ├── Stencil ref/mask front                                        │   │
│  │    ├── Stencil ref/mask back                                         │   │
│  │    └── Polygonal + backface flags                                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      TEXTURE CACHE                                   │   │
│  │                                                                      │   │
│  │  MetalTextureCache                                                   │   │
│  │    ├── Caches uploaded guest textures                                │   │
│  │    ├── Handles format conversion (Xbox formats → Metal formats)     │   │
│  │    ├── Manages texture views for different usage                     │   │
│  │    └── Tracks dirty regions for re-upload                            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                   RENDER TARGET CACHE                                │   │
│  │                                                                      │   │
│  │  MetalRenderTargetCache                                              │   │
│  │    ├── Maps EDRAM regions → MTL::Texture                            │   │
│  │    ├── Handles EDRAM aliasing (same memory, different views)        │   │
│  │    ├── Manages resolve operations (EDRAM → main memory)             │   │
│  │    └── Creates depth/stencil textures                                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    DISK CACHE (Persistent)                           │   │
│  │                                                                      │   │
│  │  shader_storage_root_/                                               │   │
│  │    ├── metallib_cache/         (Compiled .metallib files)           │   │
│  │    ├── pipeline_disk_cache     (Pipeline configurations)            │   │
│  │    └── pipeline_binary_archive (Precompiled pipeline binaries)      │   │
│  │                                                                      │   │
│  │  InitializeShaderStorage() loads on startup                          │   │
│  │  PrewarmPipelineBinaryArchive() pre-compiles pipelines               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Shared ↔ Metal Interface

```
┌────────────────────────────────────────────────────────────────────────────┐
│               ABSTRACT INTERFACE (CommandProcessor base)                    │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  VIRTUAL METHODS (must be implemented by Metal backend)              │   │
│  │                                                                      │   │
│  │  bool SetupContext() override;                                       │   │
│  │    → Initialize Metal device, command queue, caches                  │   │
│  │                                                                      │   │
│  │  void ShutdownContext() override;                                    │   │
│  │    → Release Metal resources, flush pending work                     │   │
│  │                                                                      │   │
│  │  Shader* LoadShader(type, address, host_addr, dword_count) override; │   │
│  │    → Translate Xbox microcode → Metal shader                         │   │
│  │                                                                      │   │
│  │  bool IssueDraw(prim_type, index_count, ib_info, major_mode) override│   │
│  │    → Full Metal draw pipeline                                        │   │
│  │                                                                      │   │
│  │  bool IssueCopy() override;                                          │   │
│  │    → EDRAM resolve to main memory                                    │   │
│  │                                                                      │   │
│  │  void RestoreEdramSnapshot(snapshot) override;                       │   │
│  │    → Restore EDRAM state for trace playback                          │   │
│  │                                                                      │   │
│  │  void TracePlaybackWroteMemory(base, length) override;               │   │
│  │    → Invalidate caches for trace-modified memory                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  SHARED FUNCTIONALITY (inherited from CommandProcessor)              │   │
│  │                                                                      │   │
│  │  WorkerThreadMain()      → Worker thread entry point                 │   │
│  │  ExecutePrimaryBuffer()  → Execute PM4 ring buffer commands          │   │
│  │  ExecutePacket()         → Parse and dispatch single packet          │   │
│  │  ExecutePacketType3*()   → PM4 command handlers                      │   │
│  │                                                                      │   │
│  │  register_file_          → Access to all GPU registers               │   │
│  │  active_vertex_shader_   → Currently bound vertex shader             │   │
│  │  active_pixel_shader_    → Currently bound pixel shader              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  METAL-SPECIFIC EXTENSIONS                                           │   │
│  │                                                                      │   │
│  │  MTL::Device* GetMetalDevice()                                       │   │
│  │  MTL::CommandQueue* GetMetalCommandQueue()                           │   │
│  │  MTL::CommandBuffer* EnsureCommandBuffer()                           │   │
│  │  void EndRenderEncoder()                                             │   │
│  │                                                                      │   │
│  │  MetalSharedMemory* shared_memory()                                  │   │
│  │  MetalRenderTargetCache* render_target_cache()                       │   │
│  │  MetalTextureCache* texture_cache()                                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
├────────────────────────────────────────────────────────────────────────────┤
│              ABSTRACT INTERFACE (GraphicsSystem base)                       │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  VIRTUAL METHODS                                                     │   │
│  │                                                                      │   │
│  │  std::unique_ptr<CommandProcessor> CreateCommandProcessor() override;│   │
│  │    → Returns MetalCommandProcessor instance                          │   │
│  │                                                                      │   │
│  │  void Swap() override;                                               │   │
│  │    → Present frame to screen via MetalPresenter                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  SHARED FUNCTIONALITY                                                │   │
│  │                                                                      │   │
│  │  Initialize()             → System initialization                    │   │
│  │  Shutdown()               → System shutdown                          │   │
│  │  ReadRegister()           → Guest register read                      │   │
│  │  WriteRegister()          → Guest register write                     │   │
│  │  SetInterruptCallback()   → CPU interrupt hookup                     │   │
│  │  MarkVblank()             → VBlank signal                            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Memory Flow

```
┌────────────────────────────────────────────────────────────────────────────┐
│                          MEMORY ARCHITECTURE                                │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    GUEST (Xbox 360) MEMORY                           │   │
│  │                                                                      │   │
│  │   ┌───────────────┐    ┌───────────────┐    ┌───────────────┐       │   │
│  │   │  Main Memory  │    │    EDRAM      │    │  Ring Buffer  │       │   │
│  │   │   (512 MB)    │    │   (10 MB)     │    │  (PM4 cmds)   │       │   │
│  │   │               │    │               │    │               │       │   │
│  │   │ - Textures    │    │ - Render      │    │ - Draw calls  │       │   │
│  │   │ - Vertices    │    │   targets     │    │ - State       │       │   │
│  │   │ - Indices     │    │ - Depth       │    │ - Sync        │       │   │
│  │   │ - Constants   │    │ - Stencil     │    │               │       │   │
│  │   └───────┬───────┘    └───────┬───────┘    └───────┬───────┘       │   │
│  │           │                    │                    │               │   │
│  └───────────┼────────────────────┼────────────────────┼───────────────┘   │
│              │                    │                    │                    │
│              ▼                    ▼                    ▼                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    METAL BACKEND MAPPING                             │   │
│  │                                                                      │   │
│  │   ┌───────────────┐    ┌───────────────┐    ┌───────────────┐       │   │
│  │   │MetalShared    │    │MetalRender    │    │CommandProcessor│      │   │
│  │   │Memory         │    │TargetCache    │    │RingBuffer      │      │   │
│  │   │               │    │               │    │               │       │   │
│  │   │ MTL::Buffer   │    │ MTL::Texture  │    │ reader_       │       │   │
│  │   │ (shared/      │    │ per RT/DS     │    │ (circular)    │       │   │
│  │   │  managed)     │    │               │    │               │       │   │
│  │   └───────┬───────┘    └───────┬───────┘    └───────────────┘       │   │
│  │           │                    │                                     │   │
│  │           │                    │                                     │   │
│  │           ▼                    ▼                                     │   │
│  │   ┌───────────────────────────────────────────────────────────┐     │   │
│  │   │                   DESCRIPTOR HEAPS                         │     │   │
│  │   │                                                            │     │   │
│  │   │  res_heap_ab_     Resource descriptors (SRV/UAV)          │     │   │
│  │   │  smp_heap_ab_     Sampler descriptors                      │     │   │
│  │   │  cbv_heap_ab_     Constant buffer views (b0-b4)           │     │   │
│  │   │  top_level_ab_    Root signature pointers                  │     │   │
│  │   │  uniforms_buffer_ Raw constant data                        │     │   │
│  │   │                                                            │     │   │
│  │   │  (Ring-buffered per draw to avoid GPU/CPU conflicts)      │     │   │
│  │   └───────────────────────────────────────────────────────────┘     │   │
│  │                                                                      │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 9. Swap/Present Flow

```
┌────────────────────────────────────────────────────────────────────────────┐
│                           SWAP / PRESENT FLOW                               │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Game writes frontbuffer pointer to register                               │
│           │                                                                 │
│           ▼                                                                 │
│   ┌───────────────────┐                                                     │
│   │  GraphicsSystem   │                                                     │
│   │  MarkVblank()     │                                                     │
│   └─────────┬─────────┘                                                     │
│             │                                                               │
│             ▼                                                               │
│   ┌───────────────────┐                                                     │
│   │ CommandProcessor  │                                                     │
│   │ IssueSwap()       │                                                     │
│   │ (virtual)         │                                                     │
│   └─────────┬─────────┘                                                     │
│             │                                                               │
│             ▼                                                               │
│   ┌───────────────────────────────────────────────────────────────────┐    │
│   │  MetalCommandProcessor::IssueSwap()                                │    │
│   │                                                                    │    │
│   │   1. Get frontbuffer from MetalRenderTargetCache                  │    │
│   │                                                                    │    │
│   │   2. If needed, resolve EDRAM → texture                           │    │
│   │                                                                    │    │
│   │   3. Copy/blit to swap_state_.front_buffer_texture                │    │
│   │                                                                    │    │
│   │   4. Update swap_state_ with new frame                            │    │
│   │                                                                    │    │
│   │   5. Signal frame_ready condition                                  │    │
│   └─────────┬─────────────────────────────────────────────────────────┘    │
│             │                                                               │
│             ▼                                                               │
│   ┌───────────────────────────────────────────────────────────────────┐    │
│   │  MetalPresenter (UI thread)                                        │    │
│   │                                                                    │    │
│   │   1. Wait for frame_ready                                          │    │
│   │                                                                    │    │
│   │   2. Get swap_state_.front_buffer_texture                          │    │
│   │                                                                    │    │
│   │   3. Apply post-processing (if enabled)                            │    │
│   │      - MetalFX upscaling                                           │    │
│   │      - Gamma correction                                            │    │
│   │                                                                    │    │
│   │   4. Render to CAMetalLayer drawable                               │    │
│   │                                                                    │    │
│   │   5. Present drawable                                              │    │
│   └───────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 10. EDRAM Architecture (Xbox 360 Embedded DRAM)

### 10.1 What is EDRAM?

EDRAM (Embedded DRAM) is **10 MB of high-bandwidth memory** physically embedded on the Xbox 360's Xenos GPU die. It serves as the exclusive storage for render targets and depth/stencil buffers during rendering.

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    XBOX 360 EDRAM PHYSICAL LAYOUT                          │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Total Size: 10,485,760 bytes (10 MB)                                     │
│   Organization: 2048 tiles × 5120 bytes/tile                               │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                         TILE STRUCTURE                               │  │
│   │                                                                      │  │
│   │   Each tile at 32bpp:  80 samples wide × 16 samples tall            │  │
│   │   Each tile at 64bpp:  40 samples wide × 16 samples tall            │  │
│   │                                                                      │  │
│   │   ┌────────────────────────────────────────────────────┐            │  │
│   │   │  Tile 0    │  Tile 1    │  Tile 2    │ ... │ Tile N │            │  │
│   │   │  (80×16)   │  (80×16)   │  (80×16)   │     │        │            │  │
│   │   └────────────────────────────────────────────────────┘            │  │
│   │                                                                      │  │
│   │   MSAA affects tile dimensions:                                      │  │
│   │   - 1x MSAA: 80×16 samples = 80×16 pixels                           │  │
│   │   - 2x MSAA: 80×16 samples = 40×16 pixels (2 samples/pixel)         │  │
│   │   - 4x MSAA: 80×16 samples = 40×8 pixels  (4 samples/pixel)         │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                     RENDER TARGET BINDING                            │  │
│   │                                                                      │  │
│   │   Games bind RTs to EDRAM regions via base tile offset:             │  │
│   │                                                                      │  │
│   │   RT0: base_tiles=0,   pitch=16 tiles (1280px @ 32bpp)              │  │
│   │   RT1: base_tiles=256, pitch=16 tiles                                │  │
│   │   Depth: base_tiles=512, pitch=16 tiles                              │  │
│   │                                                                      │  │
│   │   Multiple RTs can ALIAS the same EDRAM region!                      │  │
│   │   (Different formats viewing same memory = ownership transfer)       │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

### 10.2 EDRAM Emulation Paths

Xenia supports two approaches for emulating EDRAM, defined in `render_target_cache.h`:

```
┌────────────────────────────────────────────────────────────────────────────┐
│                      EDRAM EMULATION STRATEGIES                             │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  PATH A: kHostRenderTargets (Metal uses this exclusively)           │   │
│  │                                                                      │   │
│  │  ┌────────────────────────────────────────────────────────────────┐ │   │
│  │  │  Strategy: Map EDRAM regions to native GPU textures            │ │   │
│  │  │                                                                 │ │   │
│  │  │  EDRAM Tile Range          Host MTL::Texture                   │ │   │
│  │  │  ─────────────────    ──►  ──────────────────                  │ │   │
│  │  │  tiles 0-255              RT0 (1280×720 RGBA8)                 │ │   │
│  │  │  tiles 256-511            RT1 (1280×720 RGBA8)                 │ │   │
│  │  │  tiles 512-767            Depth (1280×720 D24S8)               │ │   │
│  │  │                                                                 │ │   │
│  │  │  Advantages:                                                    │ │   │
│  │  │  - Uses native GPU blending, depth/stencil testing             │ │   │
│  │  │  - High performance                                             │ │   │
│  │  │  - Leverages GPU fixed-function hardware                        │ │   │
│  │  │                                                                 │ │   │
│  │  │  Challenges:                                                    │ │   │
│  │  │  - EDRAM aliasing requires ownership transfers                  │ │   │
│  │  │  - Format conversions may lose precision                        │ │   │
│  │  │  - Some blend modes differ from Xbox behavior                   │ │   │
│  │  └────────────────────────────────────────────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  PATH B: kPixelShaderInterlock (D3D12 ROV / Vulkan FSI)             │   │
│  │                                                                      │   │
│  │  ┌────────────────────────────────────────────────────────────────┐ │   │
│  │  │  Strategy: EDRAM as raw buffer, all ops in pixel shader        │ │   │
│  │  │                                                                 │ │   │
│  │  │  ┌──────────────┐      ┌───────────────────────────────────┐  │ │   │
│  │  │  │  EDRAM as    │ ◄──► │  Pixel Shader (ROV/FSI access)     │  │ │   │
│  │  │  │  Raw Buffer  │      │  - Reads destination color         │  │ │   │
│  │  │  │  (10 MB)     │      │  - Applies blend equation          │  │ │   │
│  │  │  └──────────────┘      │  - Performs depth/stencil test     │  │ │   │
│  │  │                        │  - Writes result atomically        │  │ │   │
│  │  │                        └───────────────────────────────────┘  │ │   │
│  │  │                                                                 │ │   │
│  │  │  Backend Support:                                               │ │   │
│  │  │  - D3D12: Full ROV support (RasterizerOrderedViews)            │ │   │
│  │  │  - Vulkan: Fragment shader interlock extension                  │ │   │
│  │  │  - Metal: Has ROG (Raster Order Groups) - NOT YET IMPLEMENTED  │ │   │
│  │  │                                                                 │ │   │
│  │  │  Advantages:                                                    │ │   │
│  │  │  - 100% accurate format handling                                │ │   │
│  │  │  - No ownership transfer overhead                               │ │   │
│  │  │  - Handles all edge cases correctly                             │ │   │
│  │  │                                                                 │ │   │
│  │  │  Challenges:                                                    │ │   │
│  │  │  - Requires GPU extension support                               │ │   │
│  │  │  - Currently slower than host RTs (optimization TODO)           │ │   │
│  │  └────────────────────────────────────────────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

### 10.3 EDRAM Ownership and Aliasing

When multiple render targets reference overlapping EDRAM regions, Xenia must transfer data:

```
┌────────────────────────────────────────────────────────────────────────────┐
│                      EDRAM OWNERSHIP TRANSFER FLOW                          │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Frame N: Game renders to RT_A at EDRAM tiles 0-255 (RGBA8)               │
│                                                                             │
│        ┌─────────────────────────────────────────────────────────────┐     │
│        │  EDRAM                                                       │     │
│        │  ┌─────────────┬─────────────┬──────────────────────────┐   │     │
│        │  │ tiles 0-255 │ tiles 256+  │ ...                      │   │     │
│        │  │ (RT_A owns) │ (free)      │                          │   │     │
│        │  └─────────────┴─────────────┴──────────────────────────┘   │     │
│        └─────────────────────────────────────────────────────────────┘     │
│                                    │                                        │
│   Frame N+1: Game binds RT_B at EDRAM tiles 128-383 (RG16F)                │
│              Overlaps with RT_A!                                            │
│                                    │                                        │
│                                    ▼                                        │
│        ┌─────────────────────────────────────────────────────────────┐     │
│        │  OWNERSHIP TRANSFER REQUIRED                                 │     │
│        │                                                              │     │
│        │  1. Identify overlapping tile range: 128-255                │     │
│        │                                                              │     │
│        │  2. Read pixels from RT_A's host texture                    │     │
│        │                                                              │     │
│        │  3. Convert format: RGBA8 → RG16F                           │     │
│        │     (via transfer pixel shader)                              │     │
│        │                                                              │     │
│        │  4. Write to RT_B's host texture at correct tile offsets    │     │
│        │                                                              │     │
│        │  5. Update ownership_ranges_ map                            │     │
│        └─────────────────────────────────────────────────────────────┘     │
│                                    │                                        │
│                                    ▼                                        │
│        ┌─────────────────────────────────────────────────────────────┐     │
│        │  EDRAM (post-transfer)                                       │     │
│        │  ┌───────┬─────────────────────────┬────────────────────┐   │     │
│        │  │ 0-127 │ tiles 128-383           │ tiles 384+         │   │     │
│        │  │ RT_A  │ (RT_B owns, transferred)│ (free)             │   │     │
│        │  └───────┴─────────────────────────┴────────────────────┘   │     │
│        └─────────────────────────────────────────────────────────────┘     │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

### 10.4 EDRAM Resolve (Copy to Main Memory)

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         EDRAM RESOLVE OPERATION                             │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Triggered by: PM4_EVENT_WRITE with resolve event                         │
│   Purpose: Copy EDRAM render target contents to main memory texture        │
│                                                                             │
│   ┌──────────────────────────────────────────────────────────────────────┐ │
│   │                        RESOLVE PIPELINE                               │ │
│   │                                                                       │ │
│   │   ┌─────────────┐     ┌────────────────┐     ┌─────────────────┐    │ │
│   │   │ EDRAM/Host  │     │ Resolve Compute │     │  Main Memory    │    │ │
│   │   │ RT Texture  │ ──► │ Shader          │ ──► │  Texture        │    │ │
│   │   └─────────────┘     └────────────────┘     └─────────────────┘    │ │
│   │                              │                                       │ │
│   │                              ▼                                       │ │
│   │   ┌──────────────────────────────────────────────────────────────┐  │ │
│   │   │ Shader Selection (ResolveCopyShaderIndex):                    │  │ │
│   │   │                                                               │  │ │
│   │   │ FAST Path (format-compatible, direct copy):                   │  │ │
│   │   │   kFast32bpp1x2xMSAA  - 32-bit, 1x/2x MSAA                   │  │ │
│   │   │   kFast32bpp4xMSAA    - 32-bit, 4x MSAA                      │  │ │
│   │   │   kFast64bpp1x2xMSAA  - 64-bit, 1x/2x MSAA                   │  │ │
│   │   │   kFast64bpp4xMSAA    - 64-bit, 4x MSAA                      │  │ │
│   │   │                                                               │  │ │
│   │   │ FULL Path (format conversion required):                       │  │ │
│   │   │   kFull8bpp   - 8-bit formats                                 │  │ │
│   │   │   kFull16bpp  - 16-bit formats                                │  │ │
│   │   │   kFull32bpp  - 32-bit with conversion                        │  │ │
│   │   │   kFull64bpp  - 64-bit with conversion                        │  │ │
│   │   │   kFull128bpp - 128-bit formats                               │  │ │
│   │   └──────────────────────────────────────────────────────────────┘  │ │
│   └──────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│   ┌──────────────────────────────────────────────────────────────────────┐ │
│   │                     COMMON RESOLVE OPERATIONS                         │ │
│   │                                                                       │ │
│   │  1. MSAA Resolve: Downsample 4 samples → 1 pixel (averaging/select) │ │
│   │  2. Format Pack: Convert float → normalized, apply gamma             │ │
│   │  3. Endian Swap: Xbox big-endian → host little-endian               │ │
│   │  4. Tile Detile: EDRAM tiled layout → linear texture layout          │ │
│   │  5. Resolution Scale: Handle 2x/3x internal resolution               │ │
│   └──────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

### 10.5 EDRAM Format Handling

```
┌────────────────────────────────────────────────────────────────────────────┐
│                      EDRAM COLOR FORMAT MAPPING                             │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Xbox 360 Format          Host Format         Notes                        │
│   ─────────────────────    ──────────────────  ─────────────────────────   │
│   k_8_8_8_8               RGBA8Unorm          Direct mapping               │
│   k_8_8_8_8_GAMMA         RGBA8Unorm + sRGB   PWL gamma ≠ sRGB (approx)   │
│   k_2_10_10_10            RGB10A2Unorm        Direct mapping               │
│   k_2_10_10_10_FLOAT      RGB10A2Unorm*       Range 0-31.875 → shader     │
│   k_16_16                 RG16Snorm*          Range -32..32 → -1..1       │
│   k_16_16_16_16           RGBA16Snorm*        Range -32..32 → -1..1       │
│   k_16_16_FLOAT           RG16Float           Direct mapping               │
│   k_16_16_16_16_FLOAT     RGBA16Float         Direct mapping               │
│   k_32_FLOAT              R32Float            Direct mapping               │
│   k_32_32_FLOAT           RG32Float           Direct mapping               │
│                                                                             │
│   * = Requires shader conversion for correct range/precision               │
│                                                                             │
├────────────────────────────────────────────────────────────────────────────┤
│                      EDRAM DEPTH FORMAT MAPPING                             │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Xbox 360 Format          Host Format         Notes                        │
│   ─────────────────────    ──────────────────  ─────────────────────────   │
│   D24S8 (unorm24)         Depth24Stencil8     Direct if supported          │
│   D24FS8 (float24/20e4)   Depth32Float*       20e4 → 32f via shader       │
│                                                                             │
│   * Float24 (20e4) uses 20-bit mantissa + 4-bit exponent                   │
│     Must convert in pixel shader to avoid precision loss                    │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 11. Built-in Utility Shaders (src/xenia/gpu/shaders)

These are **pre-compiled shaders** embedded in Xenia for GPU emulation infrastructure (not guest game shaders).

### 11.1 Shader Categories

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    BUILT-IN UTILITY SHADER ARCHITECTURE                     │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  TEXTURE LOAD SHADERS (60+ variants)                                 │   │
│  │  Location: texture_load_*.cs.xesl                                    │   │
│  │                                                                      │   │
│  │  Purpose: Convert Xbox 360 texture formats to host GPU formats      │   │
│  │                                                                      │   │
│  │  ┌────────────────────────────────────────────────────────────────┐ │   │
│  │  │  Xbox 360 Texture (tiled, big-endian, DXT/CTX/etc.)            │ │   │
│  │  │              │                                                  │ │   │
│  │  │              ▼  Compute Shader (4×32 threads)                  │ │   │
│  │  │  ┌──────────────────────────────────────────────────────────┐  │ │   │
│  │  │  │  - Detile from Xbox tiled layout                          │  │ │   │
│  │  │  │  - Endian swap (big → little)                             │  │ │   │
│  │  │  │  - Decompress (DXT1/3/5, CTX1, DXN)                       │  │ │   │
│  │  │  │  - Format convert (depth, color)                          │  │ │   │
│  │  │  └──────────────────────────────────────────────────────────┘  │ │   │
│  │  │              │                                                  │ │   │
│  │  │              ▼                                                  │ │   │
│  │  │  Host GPU Texture (linear, little-endian, native format)       │ │   │
│  │  └────────────────────────────────────────────────────────────────┘ │   │
│  │                                                                      │   │
│  │  Variants by bytes-per-block: 8bpb, 16bpb, 32bpb, 64bpb, 128bpb    │   │
│  │  Scaled variants for resolution upscaling: *_scaled.cs.xesl         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  RESOLVE SHADERS (20+ variants)                                      │   │
│  │  Location: resolve_*.cs.xesl                                         │   │
│  │                                                                      │   │
│  │  Purpose: Copy EDRAM render targets to main memory                   │   │
│  │                                                                      │   │
│  │  Types:                                                              │   │
│  │  - resolve_fast_*   : Direct copy when formats match                │   │
│  │  - resolve_full_*   : Format conversion during copy                  │   │
│  │  - resolve_clear_*  : Clear EDRAM while resolving                    │   │
│  │                                                                      │   │
│  │  Used by: MetalRenderTargetCache::Resolve()                          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  EDRAM BLEND SHADERS (6 variants) [DEAD CODE - UNUSED]              │   │
│  │  Location: edram_blend_*.cs.xesl                                     │   │
│  │                                                                      │   │
│  │  ┌────────────────────────────────────────────────────────────────┐ │   │
│  │  │  WARNING: These shaders are NOT USED by ANY backend.           │ │   │
│  │  │                                                                 │ │   │
│  │  │  History: These were created for a failed experiment to        │ │   │
│  │  │  implement a compute-based ROV fallback for Metal. The         │ │   │
│  │  │  approach was abandoned, but the shaders were never removed.   │ │   │
│  │  │                                                                 │ │   │
│  │  │  Current status:                                                │ │   │
│  │  │  - Metal: Uses kHostRenderTargets only (no PSI)                │ │   │
│  │  │  - D3D12: Uses ROV directly in translated pixel shaders        │ │   │
│  │  │  - Vulkan: Uses FSI directly in translated pixel shaders       │ │   │
│  │  │                                                                 │ │   │
│  │  │  These files should be DELETED:                                 │ │   │
│  │  │  - edram_blend_32bpp_*.cs.xesl (3 files)                       │ │   │
│  │  │  - edram_blend_64bpp_*.cs.xesl (3 files)                       │ │   │
│  │  │  - edram_blend_*_common.xesli (2 files)                        │ │   │
│  │  │  - bytecode/metal/edram_blend_*.h (6 files)                    │ │   │
│  │  └────────────────────────────────────────────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  HOST DEPTH STORE SHADERS (3 variants)                               │   │
│  │  Location: host_depth_store_*xmsaa.cs.xesl                           │   │
│  │                                                                      │   │
│  │  Purpose: Convert and store depth from EDRAM/host to texture         │   │
│  │  Variants: 1x, 2x, 4x MSAA                                           │   │
│  │                                                                      │   │
│  │  Used by: MetalRenderTargetCache for depth readback                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  POST-PROCESSING SHADERS                                             │   │
│  │                                                                      │   │
│  │  apply_gamma_*.xesl   : PWL/table gamma correction                  │   │
│  │  fxaa.cs.hlsl         : Fast approximate anti-aliasing              │   │
│  │  fxaa_extreme.cs.hlsl : Higher quality FXAA                         │   │
│  │                                                                      │   │
│  │  Used by: MetalPresenter for final output processing                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

### 11.2 Shared HLSL Headers (xenos_draw.hlsli)

The shared HLSL headers define structures used by both guest shader translation and built-in tessellation shaders:

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    SHARED HLSL STRUCTURES                                   │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  xenos_draw.hlsli defines:                                                  │
│                                                                             │
│  cbuffer xe_system_cbuffer : register(b0)                                  │
│  ├── xe_flags                    - System state flags                       │
│  ├── xe_tessellation_factor_range - Tessellation limits                    │
│  ├── xe_vertex_index_*           - Index buffer parameters                  │
│  ├── xe_user_clip_planes[6]      - Clip plane equations                    │
│  ├── xe_ndc_scale/offset         - Viewport transform                      │
│  ├── xe_point_*                  - Point sprite parameters                  │
│  ├── xe_texture_swizzled_signs   - Texture format info                     │
│  ├── xe_edram_*                  - EDRAM addressing (for ROV path)         │
│  └── xe_color_exp_bias           - HDR color scaling                       │
│                                                                             │
│  Tessellation Structures:                                                   │
│  ├── XeHSControlPointInputIndexed  - Indexed tessellation input            │
│  ├── XeHSControlPointInputAdaptive - Adaptive tessellation input           │
│  ├── XeHSControlPointOutput        - Hull shader output                    │
│  ├── XeVertexPrePS                 - VS → PS interpolators                 │
│  ├── XeVertexPostGS                - Post-geometry shader vertex           │
│  └── XeVertexPreGS                 - Pre-geometry shader vertex            │
│                                                                             │
│  These headers are included by:                                             │
│  1. DxbcShaderTranslator - when generating translated guest shaders        │
│  2. tessellation_*.hlsl  - built-in tessellation shaders                   │
│  3. continuous_*.hs.hlsl / discrete_*.hs.hlsl - hull shaders               │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

### 11.3 Shader Compilation Pipeline

```
┌────────────────────────────────────────────────────────────────────────────┐
│                   UTILITY SHADER BUILD PIPELINE                             │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   SOURCE                    COMPILE TIME                 RUNTIME            │
│   ──────                    ────────────                 ───────            │
│                                                                             │
│   ┌────────────┐           ┌────────────┐           ┌────────────┐         │
│   │  .xesl     │           │  HLSL/DXC  │           │  Bytecode  │         │
│   │  .hlsl     │ ────────► │  Compiler  │ ────────► │  Headers   │         │
│   │  source    │           │            │           │  (.h)      │         │
│   └────────────┘           └────────────┘           └────────────┘         │
│                                   │                        │                │
│                                   ▼                        │                │
│                            ┌────────────┐                  │                │
│                            │ Per-backend│                  │                │
│                            │ compilation│                  │                │
│                            └────────────┘                  │                │
│                                   │                        │                │
│            ┌──────────────────────┼──────────────────────┐ │                │
│            │                      │                      │ │                │
│            ▼                      ▼                      ▼ ▼                │
│   ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐          │
│   │ bytecode/metal/ │   │ bytecode/vulkan │   │ bytecode/d3d12/ │          │
│   │                 │   │ _spirv/         │   │                 │          │
│   │ *_cs.h          │   │ *_cs.h          │   │ *_cs.h          │          │
│   │ (metallib)      │   │ (SPIR-V)        │   │ (DXBC)          │          │
│   └────────┬────────┘   └─────────────────┘   └─────────────────┘          │
│            │                                                                │
│            ▼                                                                │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  METAL RUNTIME LOADING                                               │  │
│   │                                                                      │  │
│   │  #include "bytecode/metal/resolve_full_32bpp_cs.h"                  │  │
│   │                                                                      │  │
│   │  MTL::Library* lib = device->newLibrary(                            │  │
│   │      dispatch_data_create(resolve_full_32bpp_cs_metallib, size));   │  │
│   │                                                                      │  │
│   │  MTL::ComputePipelineState* pipeline =                              │  │
│   │      device->newComputePipelineState(lib->newFunction("main"));     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

### 11.4 Integration with Draw Pipeline

```
┌────────────────────────────────────────────────────────────────────────────┐
│              UTILITY SHADERS IN THE RENDERING PIPELINE                      │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  TEXTURE LOADING (before draw)                                       │  │
│   │                                                                      │  │
│   │  MetalTextureCache::RequestTextures()                                │  │
│   │       │                                                              │  │
│   │       ▼                                                              │  │
│   │  For each texture fetch constant:                                    │  │
│   │       │                                                              │  │
│   │       ├── Check if already cached                                    │  │
│   │       │                                                              │  │
│   │       └── If not: dispatch texture_load_*bpb compute shader         │  │
│   │                   ┌────────────────────────────────────────┐        │  │
│   │                   │  Input: Guest memory buffer            │        │  │
│   │                   │  Output: Host MTL::Texture             │        │  │
│   │                   │  Work: Detile + endian + decompress    │        │  │
│   │                   └────────────────────────────────────────┘        │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  RENDER TARGET OWNERSHIP TRANSFER (before draw if aliasing)          │  │
│   │                                                                      │  │
│   │  MetalRenderTargetCache::Update()                                    │  │
│   │       │                                                              │  │
│   │       ▼                                                              │  │
│   │  If ownership_ranges_ shows overlap:                                 │  │
│   │       │                                                              │  │
│   │       └── Dispatch transfer_ps fragment shader (embedded MSL)       │  │
│   │                   ┌────────────────────────────────────────┐        │  │
│   │                   │  Input: Previous owner's texture       │        │  │
│   │                   │  Output: New owner's texture           │        │  │
│   │                   │  Work: Format convert + tile remap     │        │  │
│   │                   └────────────────────────────────────────┘        │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  ACTUAL DRAW (game's translated shaders)                             │  │
│   │                                                                      │  │
│   │  MetalCommandProcessor::IssueDraw()                                  │  │
│   │       │                                                              │  │
│   │       └── Uses guest shaders (Xbox → DXBC → DXIL → Metal IR)        │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  RESOLVE (after draw, triggered by game)                             │  │
│   │                                                                      │  │
│   │  MetalRenderTargetCache::Resolve()                                   │  │
│   │       │                                                              │  │
│   │       └── Dispatch resolve_*bpp compute shader                       │  │
│   │                   ┌────────────────────────────────────────┐        │  │
│   │                   │  Input: EDRAM/Host RT texture          │        │  │
│   │                   │  Output: Main memory texture           │        │  │
│   │                   │  Work: MSAA resolve + format pack      │        │  │
│   │                   └────────────────────────────────────────┘        │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  PRESENT (after frame complete)                                      │  │
│   │                                                                      │  │
│   │  MetalPresenter::Present()                                           │  │
│   │       │                                                              │  │
│   │       ├── Optional: apply_gamma_* shader                             │  │
│   │       ├── Optional: fxaa.cs shader                                   │  │
│   │       └── Blit to CAMetalLayer drawable                              │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 12. Metal-Embedded Shaders (Inline MSL)

The Metal backend contains **~7000+ lines of inline Metal Shading Language** for EDRAM operations that cannot use the pre-compiled XeSL shaders.

### 12.1 Embedded Shader Categories

```
┌────────────────────────────────────────────────────────────────────────────┐
│              METAL-EMBEDDED SHADERS (metal_render_target_cache.cc)          │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  EDRAM DUMP COMPUTE KERNELS (9 variants)                             │   │
│  │  Purpose: Copy host render target data TO EDRAM buffer               │   │
│  │                                                                      │   │
│  │  Variants:                                                           │   │
│  │  - edram_dump_color_32bpp_1xmsaa / 2xmsaa / 4xmsaa                  │   │
│  │  - edram_dump_color_64bpp_1xmsaa / 2xmsaa / 4xmsaa                  │   │
│  │  - edram_dump_depth_32bpp_1xmsaa / 2xmsaa / 4xmsaa                  │   │
│  │                                                                      │   │
│  │  ┌──────────────────────────────────────────────────────────────┐   │   │
│  │  │  kernel void edram_dump_color_32bpp(                          │   │   │
│  │  │      texture2d<float> source [[texture(0)]],                  │   │   │
│  │  │      device uint* edram [[buffer(0)]],                        │   │   │
│  │  │      constant DumpParams& params [[buffer(1)]],               │   │   │
│  │  │      uint2 tid [[thread_position_in_grid]])                   │   │   │
│  │  │  {                                                            │   │   │
│  │  │      // Calculate EDRAM tile offset (Xbox tiled layout)       │   │   │
│  │  │      uint offset = XeEdramOffset(tid, params);                │   │   │
│  │  │      // Read from host texture, pack to Xbox format           │   │   │
│  │  │      float4 color = source.read(tid);                         │   │   │
│  │  │      edram[offset] = pack_float4_to_rgba8(color);             │   │   │
│  │  │  }                                                            │   │   │
│  │  └──────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  EDRAM LOAD FRAGMENT SHADERS                                         │   │
│  │  Purpose: Load EDRAM buffer data INTO host render target             │   │
│  │                                                                      │   │
│  │  Components:                                                         │   │
│  │  - edram_load_vs: Fullscreen quad vertex shader                     │   │
│  │  - edram_load_ps: Fragment shader with format unpacking             │   │
│  │                                                                      │   │
│  │  ┌──────────────────────────────────────────────────────────────┐   │   │
│  │  │  fragment float4 edram_load_ps(                               │   │   │
│  │  │      float4 position [[position]],                            │   │   │
│  │  │      device const uint* edram [[buffer(0)]],                  │   │   │
│  │  │      constant LoadParams& params [[buffer(1)]])               │   │   │
│  │  │  {                                                            │   │   │
│  │  │      uint2 pixel = uint2(position.xy);                        │   │   │
│  │  │      uint offset = XeEdramOffset(pixel, params);              │   │   │
│  │  │      uint packed = edram[offset];                             │   │   │
│  │  │      return XeUnpackR8G8B8A8UNorm(packed);                    │   │   │
│  │  │  }                                                            │   │   │
│  │  └──────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  OWNERSHIP TRANSFER SHADERS (massive, ~1400 lines)                   │   │
│  │  Purpose: Copy and convert between aliased EDRAM regions             │   │
│  │                                                                      │   │
│  │  Components:                                                         │   │
│  │  - transfer_vs: Fullscreen quad                                      │   │
│  │  - transfer_ps: Complex format conversion logic                      │   │
│  │                                                                      │   │
│  │  Handles:                                                            │   │
│  │  - Source/dest tile coordinate mapping                               │   │
│  │  - MSAA sample remapping (1x↔2x↔4x)                                  │   │
│  │  - 32bpp ↔ 64bpp format width adjustment                            │   │
│  │  - Color ↔ depth type conversion                                    │   │
│  │  - Conditional stencil operations                                    │   │
│  │                                                                      │   │
│  │  Compile-time specialization via macros:                             │   │
│  │  - XE_TRANSFER_SOURCE_IS_COLOR                                       │   │
│  │  - XE_TRANSFER_SOURCE_IS_MULTISAMPLE                                 │   │
│  │  - XE_TRANSFER_HAS_HOST_DEPTH                                        │   │
│  │  - XE_TRANSFER_OUTPUT_COLOR                                          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  TRANSFER CLEAR SHADERS                                              │   │
│  │  Purpose: Clear EDRAM regions during ownership change                │   │
│  │                                                                      │   │
│  │  - transfer_clear_vs: Fullscreen quad                                │   │
│  │  - transfer_clear_color_float_ps: Float color clear                  │   │
│  │  - transfer_clear_color_uint_ps: Uint color clear                    │   │
│  │  - transfer_clear_depth_ps: Depth buffer clear                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  DEPTH READBACK COMPUTE KERNELS                                      │   │
│  │  Purpose: Read depth texture to CPU-accessible buffer                │   │
│  │                                                                      │   │
│  │  - readback_depth: Single-sample depth                               │   │
│  │  - readback_depth_ms: Multisample depth with sample selection        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

### 12.2 Why Embedded vs Pre-compiled?

```
┌────────────────────────────────────────────────────────────────────────────┐
│           EMBEDDED MSL vs PRE-COMPILED XESL SHADER COMPARISON              │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   PRE-COMPILED (src/xenia/gpu/shaders/bytecode/metal/)                     │
│   ──────────────────────────────────────────────────────                   │
│   - Cross-platform XeSL source                                              │
│   - Compiled offline by build system                                        │
│   - Embedded as static byte arrays                                          │
│   - Used for: texture_load, resolve, host_depth_store                       │
│   - Advantage: Consistent across backends                                   │
│                                                                             │
│   EMBEDDED MSL (inline in metal_render_target_cache.cc)                     │
│   ─────────────────────────────────────────────────────                     │
│   - Metal-only MSL source strings                                           │
│   - Compiled at runtime via device->newLibrary(sourceString)               │
│   - Used for: EDRAM dump/load, ownership transfer, clears                   │
│   - Reason: Metal-specific optimizations, dynamic specialization            │
│                                                                             │
│   ┌──────────────────────────────────────────────────────────────────────┐ │
│   │  Transfer shaders MUST be embedded because:                           │ │
│   │                                                                       │ │
│   │  1. Highly dynamic configuration (dozens of macro combinations)      │ │
│   │  2. Pipeline created on-demand based on actual RT configurations     │ │
│   │  3. Would require 100+ pre-compiled variants otherwise               │ │
│   │  4. Metal's fast runtime compilation makes this practical            │ │
│   └──────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 13. File Structure Reference

```
src/xenia/gpu/
├── graphics_system.h/cc          # Base graphics system
├── command_processor.h/cc        # Base command processor
├── pm4_command_processor_*.h     # PM4 packet handlers (shared)
├── register_file.h/cc            # GPU register storage
├── register_table.inc            # Register definitions (20K+)
├── registers.h                   # Register structure types
├── xenos.h                       # Xenos GPU constants (EDRAM sizes, etc.)
├── shader.h                      # Base shader abstraction
├── draw_util.h                   # Draw state utilities
├── primitive_processor.h         # Primitive conversion (shared)
├── render_target_cache.h/cc      # Base render target cache + EDRAM logic
├── dxbc_shader_translator.h/cc   # Xbox → DXBC translation
│
├── shaders/                      # Built-in utility shaders (XeSL/HLSL)
│   ├── *.xesl, *.hlsl           # Source shaders
│   ├── texture_load_*.cs.xesl   # Texture format conversion (60+)
│   ├── resolve_*.cs.xesl        # EDRAM resolve shaders (20+)
│   ├── edram_blend_*.cs.xesl    # [DEAD CODE - unused by all backends]
│   ├── host_depth_store_*.xesl  # Depth readback
│   ├── apply_gamma_*.xesl       # Gamma correction
│   │
│   ├── # Shared HLSL for tessellation (used via DXBC→DXIL→Metal pipeline)
│   ├── xenos_draw.hlsli         # System constants, VS/PS/GS structures
│   ├── tessellation_*.vs.hlsl   # Tessellation vertex shaders
│   ├── *_quad*.hs.hlsl          # Quad tessellation hull shaders
│   ├── *_triangle*.hs.hlsl      # Triangle tessellation hull shaders
│   ├── fxaa*.hlsl               # Anti-aliasing
│   │
│   ├── edram.xesli              # EDRAM addressing helpers
│   ├── pixel_formats.xesli      # Format pack/unpack
│   ├── texture_address.xesli    # Texture tiling math
│   │
│   └── bytecode/                # Pre-compiled shader bytecode
│       ├── metal/               # Metal .metallib bytecode headers
│       ├── vulkan_spirv/        # Vulkan SPIR-V bytecode
│       └── d3d12_5_1/           # D3D12 DXBC bytecode
│
├── d3d12/                        # D3D12 backend (REFERENCE IMPLEMENTATION)
│   ├── d3d12_graphics_system.h/cc
│   ├── d3d12_command_processor.h/cc
│   ├── d3d12_render_target_cache.h/cc  # Full ROV support
│   ├── d3d12_texture_cache.h/cc
│   ├── d3d12_shared_memory.h/cc
│   ├── pipeline_cache.h/cc             # Shader pipeline management
│   └── deferred_command_list.h/cc
│
└── metal/                        # Metal backend
    ├── metal_graphics_system.h/cc
    ├── metal_command_processor.h/cc
    ├── metal_shared_memory.h/cc
    ├── metal_texture_cache.h/cc
    ├── metal_render_target_cache.h/cc  # ~7000 lines embedded MSL shaders
    ├── metal_shader.h/cc
    ├── metal_shader_converter.h/cc     # DXIL → Metal IR (Apple MSC)
    ├── dxbc_to_dxil_converter.h/cc     # DXBC → DXIL
    ├── metal_primitive_processor.h/cc
    ├── metal_geometry_shader.h/cc
    └── metal_resource_tracker.h/cc
```

---

## 14. Summary

The Metal graphics backend follows a layered architecture with distinct shader categories:

### Architecture Layers

1. **Shared Layer** (`src/xenia/gpu`): Provides PM4 command parsing, register management, EDRAM emulation logic, and defines the abstract interface that all backends must implement.

2. **Metal Layer** (`src/xenia/gpu/metal`): Implements the abstract interface using Apple's Metal API, including shader translation (Xbox → DXBC → DXIL → Metal IR), resource caching, and command encoding.

3. **State Flow**: Commands flow from the game's ring buffer through the shared command processor, which parses PM4 packets and calls backend-specific virtual methods (IssueDraw, IssueCopy, LoadShader) for actual GPU work.

4. **Caching**: Multiple cache layers (shaders, pipelines, textures, render targets) minimize redundant work and enable persistent storage for faster subsequent launches.

### Shader Categories

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    THREE TYPES OF SHADERS IN XENIA                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. GUEST SHADERS (dynamic, per-game)                                   │
│     Source: Game's Xbox 360 microcode                                   │
│     Translation: Xbox → DXBC → DXIL → Metal IR (via Apple MSC)          │
│     Usage: Actual game rendering (vertex/pixel shaders)                 │
│     Includes: xenos_draw.hlsli system constants                         │
│                                                                          │
│  2. BUILT-IN UTILITY SHADERS (src/xenia/gpu/shaders/)                   │
│     Source: XeSL/HLSL, pre-compiled to bytecode                         │
│     Purpose: GPU emulation infrastructure                                │
│     Examples: texture_load_*, resolve_*, host_depth_store_*             │
│     Tessellation: *.hs.hlsl, tessellation_*.vs.hlsl (shared HLSL)       │
│                                                                          │
│     Note: edram_blend_* shaders are DEAD CODE (unused by all backends) │
│                                                                          │
│  3. METAL-EMBEDDED SHADERS (inline MSL in .cc files)                    │
│     Source: MSL strings compiled at runtime                              │
│     Purpose: Metal-specific EDRAM operations                             │
│     Examples: transfer_*, edram_load_*, edram_dump_*                    │
│     Location: metal_render_target_cache.cc (~7000 lines)                │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### EDRAM Emulation

Metal uses the **kHostRenderTargets** path exclusively:
- EDRAM regions mapped to native MTL::Texture objects
- Ownership transfers handle aliasing between render targets
- Native GPU blending (no software blending shaders)
- Resolve operations copy EDRAM content to main memory

**Future: Metal ROG Support**

Metal has **Raster Order Groups (ROG)** which is equivalent to D3D12's ROV and Vulkan's FSI. This would enable the kPixelShaderInterlock path for more accurate EDRAM emulation, but is **not yet implemented** in Xenia's Metal backend.

### D3D12 as Reference Implementation

The **D3D12 backend** (`src/xenia/gpu/d3d12/`) is the feature-complete reference that Metal is targeting parity with:

- Full ROV (RasterizerOrderedViews) support for kPixelShaderInterlock path
- Complete tessellation pipeline using shared HLSL shaders
- Geometry shader support via DXBC generation
- Pipeline state caching with binary archives

The D3D12 backend demonstrates how PSI mode integrates blending directly into translated pixel shaders rather than using separate compute shaders.

### Dead Code: edram_blend_* Shaders

The `edram_blend_*.cs.xesl` shaders and their Metal bytecode are **dead code** from a failed experiment. They were intended for a compute-based ROV fallback but the approach was abandoned. These shaders are not used by any backend and should be deleted.

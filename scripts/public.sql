/*
 Navicat Premium Dump SQL

 Source Server         : postgres
 Source Server Type    : PostgreSQL
 Source Server Version : 170006 (170006)
 Source Host           : localhost:5432
 Source Catalog        : WeKnora
 Source Schema         : public

 Target Server Type    : PostgreSQL
 Target Server Version : 170006 (170006)
 File Encoding         : 65001

 Date: 31/01/2026 12:05:39
*/


-- ----------------------------
-- Type structure for box2d
-- ----------------------------
DROP TYPE IF EXISTS "public"."box2d";
CREATE TYPE "public"."box2d" (
  INPUT = "public"."box2d_in",
  OUTPUT = "public"."box2d_out",
  INTERNALLENGTH = 65,
  CATEGORY = U,
  DELIMITER = ','
);
ALTER TYPE "public"."box2d" OWNER TO "postgres";

-- ----------------------------
-- Type structure for box2df
-- ----------------------------
DROP TYPE IF EXISTS "public"."box2df";
CREATE TYPE "public"."box2df" (
  INPUT = "public"."box2df_in",
  OUTPUT = "public"."box2df_out",
  INTERNALLENGTH = 16,
  ALIGNMENT = double,
  CATEGORY = U,
  DELIMITER = ','
);
ALTER TYPE "public"."box2df" OWNER TO "postgres";

-- ----------------------------
-- Type structure for box3d
-- ----------------------------
DROP TYPE IF EXISTS "public"."box3d";
CREATE TYPE "public"."box3d" (
  INPUT = "public"."box3d_in",
  OUTPUT = "public"."box3d_out",
  INTERNALLENGTH = 52,
  ALIGNMENT = double,
  CATEGORY = U,
  DELIMITER = ','
);
ALTER TYPE "public"."box3d" OWNER TO "postgres";

-- ----------------------------
-- Type structure for geography
-- ----------------------------
DROP TYPE IF EXISTS "public"."geography";
CREATE TYPE "public"."geography" (
  INPUT = "public"."geography_in",
  OUTPUT = "public"."geography_out",
  RECEIVE = "public"."geography_recv",
  SEND = "public"."geography_send",
  TYPMOD_IN = "public"."geography_typmod_in",
  TYPMOD_OUT = "public"."geography_typmod_out",
  ANALYZE = "public"."geography_analyze",
  INTERNALLENGTH = VARIABLE,
  ALIGNMENT = double,
  STORAGE = main,
  CATEGORY = U,
  DELIMITER = ':'
);
ALTER TYPE "public"."geography" OWNER TO "postgres";

-- ----------------------------
-- Type structure for geometry
-- ----------------------------
DROP TYPE IF EXISTS "public"."geometry";
CREATE TYPE "public"."geometry" (
  INPUT = "public"."geometry_in",
  OUTPUT = "public"."geometry_out",
  RECEIVE = "public"."geometry_recv",
  SEND = "public"."geometry_send",
  TYPMOD_IN = "public"."geometry_typmod_in",
  TYPMOD_OUT = "public"."geometry_typmod_out",
  ANALYZE = "public"."geometry_analyze",
  INTERNALLENGTH = VARIABLE,
  ALIGNMENT = double,
  STORAGE = main,
  CATEGORY = U,
  DELIMITER = ':'
);
ALTER TYPE "public"."geometry" OWNER TO "postgres";

-- ----------------------------
-- Type structure for geometry_dump
-- ----------------------------
DROP TYPE IF EXISTS "public"."geometry_dump";
CREATE TYPE "public"."geometry_dump" AS (
  "path" int4[],
  "geom" "public"."geometry"
);
ALTER TYPE "public"."geometry_dump" OWNER TO "postgres";

-- ----------------------------
-- Type structure for gidx
-- ----------------------------
DROP TYPE IF EXISTS "public"."gidx";
CREATE TYPE "public"."gidx" (
  INPUT = "public"."gidx_in",
  OUTPUT = "public"."gidx_out",
  INTERNALLENGTH = VARIABLE,
  ALIGNMENT = double,
  CATEGORY = U,
  DELIMITER = ','
);
ALTER TYPE "public"."gidx" OWNER TO "postgres";

-- ----------------------------
-- Type structure for gtrgm
-- ----------------------------
DROP TYPE IF EXISTS "public"."gtrgm";
CREATE TYPE "public"."gtrgm" (
  INPUT = "public"."gtrgm_in",
  OUTPUT = "public"."gtrgm_out",
  INTERNALLENGTH = VARIABLE,
  CATEGORY = U,
  DELIMITER = ','
);
ALTER TYPE "public"."gtrgm" OWNER TO "postgres";

-- ----------------------------
-- Type structure for halfvec
-- ----------------------------
DROP TYPE IF EXISTS "public"."halfvec";
CREATE TYPE "public"."halfvec" (
  INPUT = "public"."halfvec_in",
  OUTPUT = "public"."halfvec_out",
  RECEIVE = "public"."halfvec_recv",
  SEND = "public"."halfvec_send",
  TYPMOD_IN = "public"."halfvec_typmod_in",
  INTERNALLENGTH = VARIABLE,
  STORAGE = external,
  CATEGORY = U,
  DELIMITER = ','
);
ALTER TYPE "public"."halfvec" OWNER TO "postgres";

-- ----------------------------
-- Type structure for sparsevec
-- ----------------------------
DROP TYPE IF EXISTS "public"."sparsevec";
CREATE TYPE "public"."sparsevec" (
  INPUT = "public"."sparsevec_in",
  OUTPUT = "public"."sparsevec_out",
  RECEIVE = "public"."sparsevec_recv",
  SEND = "public"."sparsevec_send",
  TYPMOD_IN = "public"."sparsevec_typmod_in",
  INTERNALLENGTH = VARIABLE,
  STORAGE = external,
  CATEGORY = U,
  DELIMITER = ','
);
ALTER TYPE "public"."sparsevec" OWNER TO "postgres";

-- ----------------------------
-- Type structure for spheroid
-- ----------------------------
DROP TYPE IF EXISTS "public"."spheroid";
CREATE TYPE "public"."spheroid" (
  INPUT = "public"."spheroid_in",
  OUTPUT = "public"."spheroid_out",
  INTERNALLENGTH = 65,
  ALIGNMENT = double,
  CATEGORY = U,
  DELIMITER = ','
);
ALTER TYPE "public"."spheroid" OWNER TO "postgres";

-- ----------------------------
-- Type structure for valid_detail
-- ----------------------------
DROP TYPE IF EXISTS "public"."valid_detail";
CREATE TYPE "public"."valid_detail" AS (
  "valid" bool,
  "reason" varchar COLLATE "pg_catalog"."default",
  "location" "public"."geometry"
);
ALTER TYPE "public"."valid_detail" OWNER TO "postgres";

-- ----------------------------
-- Type structure for vector
-- ----------------------------
DROP TYPE IF EXISTS "public"."vector";
CREATE TYPE "public"."vector" (
  INPUT = "public"."vector_in",
  OUTPUT = "public"."vector_out",
  RECEIVE = "public"."vector_recv",
  SEND = "public"."vector_send",
  TYPMOD_IN = "public"."vector_typmod_in",
  INTERNALLENGTH = VARIABLE,
  STORAGE = external,
  CATEGORY = U,
  DELIMITER = ','
);
ALTER TYPE "public"."vector" OWNER TO "postgres";

-- ----------------------------
-- Sequence structure for embeddings_id_seq
-- ----------------------------
DROP SEQUENCE IF EXISTS "public"."embeddings_id_seq";
CREATE SEQUENCE "public"."embeddings_id_seq" 
INCREMENT 1
MINVALUE  1
MAXVALUE 2147483647
START 1
CACHE 1;
ALTER SEQUENCE "public"."embeddings_id_seq" OWNER TO "postgres";

-- ----------------------------
-- Sequence structure for tenants_id_seq
-- ----------------------------
DROP SEQUENCE IF EXISTS "public"."tenants_id_seq";
CREATE SEQUENCE "public"."tenants_id_seq" 
INCREMENT 1
MINVALUE  1
MAXVALUE 2147483647
START 1
CACHE 1;
ALTER SEQUENCE "public"."tenants_id_seq" OWNER TO "postgres";

-- ----------------------------
-- Table structure for auth_tokens
-- ----------------------------
DROP TABLE IF EXISTS "public"."auth_tokens";
CREATE TABLE "public"."auth_tokens" (
  "id" varchar(36) COLLATE "pg_catalog"."default" NOT NULL DEFAULT uuid_generate_v4(),
  "user_id" varchar(36) COLLATE "pg_catalog"."default" NOT NULL,
  "token" text COLLATE "pg_catalog"."default" NOT NULL,
  "token_type" varchar(50) COLLATE "pg_catalog"."default" NOT NULL,
  "expires_at" timestamptz(6) NOT NULL,
  "is_revoked" bool NOT NULL DEFAULT false,
  "created_at" timestamptz(6) DEFAULT CURRENT_TIMESTAMP,
  "updated_at" timestamptz(6) DEFAULT CURRENT_TIMESTAMP
)
;
ALTER TABLE "public"."auth_tokens" OWNER TO "postgres";
COMMENT ON COLUMN "public"."auth_tokens"."id" IS 'Unique identifier of the token';
COMMENT ON COLUMN "public"."auth_tokens"."user_id" IS 'User ID that owns this token';
COMMENT ON COLUMN "public"."auth_tokens"."token" IS 'Token value (JWT or other format)';
COMMENT ON COLUMN "public"."auth_tokens"."token_type" IS 'Token type (access_token, refresh_token)';
COMMENT ON COLUMN "public"."auth_tokens"."expires_at" IS 'Token expiration time';
COMMENT ON COLUMN "public"."auth_tokens"."is_revoked" IS 'Whether the token is revoked';
COMMENT ON TABLE "public"."auth_tokens" IS 'Authentication tokens for users';

-- ----------------------------
-- Table structure for chunks
-- ----------------------------
DROP TABLE IF EXISTS "public"."chunks";
CREATE TABLE "public"."chunks" (
  "id" varchar(36) COLLATE "pg_catalog"."default" NOT NULL DEFAULT uuid_generate_v4(),
  "tenant_id" int4 NOT NULL,
  "knowledge_base_id" varchar(36) COLLATE "pg_catalog"."default" NOT NULL,
  "knowledge_id" varchar(36) COLLATE "pg_catalog"."default" NOT NULL,
  "content" text COLLATE "pg_catalog"."default" NOT NULL,
  "chunk_index" int4 NOT NULL,
  "is_enabled" bool NOT NULL DEFAULT true,
  "start_at" int4 NOT NULL,
  "end_at" int4 NOT NULL,
  "pre_chunk_id" varchar(36) COLLATE "pg_catalog"."default",
  "next_chunk_id" varchar(36) COLLATE "pg_catalog"."default",
  "chunk_type" varchar(20) COLLATE "pg_catalog"."default" NOT NULL DEFAULT 'text'::character varying,
  "parent_chunk_id" varchar(36) COLLATE "pg_catalog"."default",
  "image_info" text COLLATE "pg_catalog"."default",
  "relation_chunks" jsonb,
  "indirect_relation_chunks" jsonb,
  "created_at" timestamptz(6) DEFAULT CURRENT_TIMESTAMP,
  "updated_at" timestamptz(6) DEFAULT CURRENT_TIMESTAMP,
  "deleted_at" timestamptz(6),
  "metadata" jsonb,
  "tag_id" varchar(36) COLLATE "pg_catalog"."default",
  "status" int4 NOT NULL DEFAULT 0,
  "content_hash" varchar(64) COLLATE "pg_catalog"."default",
  "flags" int4 NOT NULL DEFAULT 1
)
;
ALTER TABLE "public"."chunks" OWNER TO "postgres";

-- ----------------------------
-- Table structure for custom_agents
-- ----------------------------
DROP TABLE IF EXISTS "public"."custom_agents";
CREATE TABLE "public"."custom_agents" (
  "id" varchar(36) COLLATE "pg_catalog"."default" NOT NULL DEFAULT uuid_generate_v4(),
  "name" varchar(255) COLLATE "pg_catalog"."default" NOT NULL,
  "description" text COLLATE "pg_catalog"."default",
  "avatar" varchar(64) COLLATE "pg_catalog"."default",
  "is_builtin" bool NOT NULL DEFAULT false,
  "tenant_id" int4 NOT NULL,
  "created_by" varchar(36) COLLATE "pg_catalog"."default",
  "config" jsonb NOT NULL DEFAULT '{}'::jsonb,
  "created_at" timestamptz(6) DEFAULT CURRENT_TIMESTAMP,
  "updated_at" timestamptz(6) DEFAULT CURRENT_TIMESTAMP,
  "deleted_at" timestamptz(6)
)
;
ALTER TABLE "public"."custom_agents" OWNER TO "postgres";

-- ----------------------------
-- Table structure for embeddings
-- ----------------------------
DROP TABLE IF EXISTS "public"."embeddings";
CREATE TABLE "public"."embeddings" (
  "id" int4 NOT NULL DEFAULT nextval('embeddings_id_seq'::regclass),
  "created_at" timestamptz(6) DEFAULT CURRENT_TIMESTAMP,
  "updated_at" timestamptz(6) DEFAULT CURRENT_TIMESTAMP,
  "source_id" varchar(64) COLLATE "pg_catalog"."default" NOT NULL,
  "source_type" int4 NOT NULL,
  "chunk_id" varchar(64) COLLATE "pg_catalog"."default",
  "knowledge_id" varchar(64) COLLATE "pg_catalog"."default",
  "knowledge_base_id" varchar(64) COLLATE "pg_catalog"."default",
  "content" text COLLATE "pg_catalog"."default",
  "dimension" int4 NOT NULL,
  "embedding" "public"."halfvec",
  "is_enabled" bool DEFAULT true,
  "tag_id" varchar(36) COLLATE "pg_catalog"."default"
)
;
ALTER TABLE "public"."embeddings" OWNER TO "postgres";

-- ----------------------------
-- Table structure for knowledge_bases
-- ----------------------------
DROP TABLE IF EXISTS "public"."knowledge_bases";
CREATE TABLE "public"."knowledge_bases" (
  "id" varchar(36) COLLATE "pg_catalog"."default" NOT NULL DEFAULT uuid_generate_v4(),
  "name" varchar(255) COLLATE "pg_catalog"."default" NOT NULL,
  "description" text COLLATE "pg_catalog"."default",
  "tenant_id" int4 NOT NULL,
  "chunking_config" jsonb NOT NULL DEFAULT '{"chunk_size": 512, "chunk_overlap": 50, "split_markers": ["\n\n", "\n", "。"], "keep_separator": true}'::jsonb,
  "image_processing_config" jsonb NOT NULL DEFAULT '{"model_id": "", "enable_multimodal": false}'::jsonb,
  "embedding_model_id" varchar(64) COLLATE "pg_catalog"."default" NOT NULL,
  "summary_model_id" varchar(64) COLLATE "pg_catalog"."default" NOT NULL,
  "cos_config" jsonb NOT NULL DEFAULT '{}'::jsonb,
  "vlm_config" jsonb NOT NULL DEFAULT '{}'::jsonb,
  "extract_config" jsonb,
  "created_at" timestamptz(6) DEFAULT CURRENT_TIMESTAMP,
  "updated_at" timestamptz(6) DEFAULT CURRENT_TIMESTAMP,
  "deleted_at" timestamptz(6),
  "is_temporary" bool NOT NULL DEFAULT false,
  "type" varchar(32) COLLATE "pg_catalog"."default" NOT NULL DEFAULT 'document'::character varying,
  "faq_config" jsonb,
  "question_generation_config" jsonb
)
;
ALTER TABLE "public"."knowledge_bases" OWNER TO "postgres";
COMMENT ON COLUMN "public"."knowledge_bases"."is_temporary" IS 'Whether this knowledge base is temporary (ephemeral) and should be hidden from UI';

-- ----------------------------
-- Table structure for knowledge_tags
-- ----------------------------
DROP TABLE IF EXISTS "public"."knowledge_tags";
CREATE TABLE "public"."knowledge_tags" (
  "id" varchar(36) COLLATE "pg_catalog"."default" NOT NULL,
  "tenant_id" int4 NOT NULL,
  "knowledge_base_id" varchar(36) COLLATE "pg_catalog"."default" NOT NULL,
  "name" varchar(128) COLLATE "pg_catalog"."default" NOT NULL,
  "color" varchar(32) COLLATE "pg_catalog"."default",
  "sort_order" int4 NOT NULL DEFAULT 0,
  "created_at" timestamptz(6) DEFAULT CURRENT_TIMESTAMP,
  "updated_at" timestamptz(6) DEFAULT CURRENT_TIMESTAMP,
  "deleted_at" timestamptz(6)
)
;
ALTER TABLE "public"."knowledge_tags" OWNER TO "postgres";

-- ----------------------------
-- Table structure for knowledges
-- ----------------------------
DROP TABLE IF EXISTS "public"."knowledges";
CREATE TABLE "public"."knowledges" (
  "id" varchar(36) COLLATE "pg_catalog"."default" NOT NULL DEFAULT uuid_generate_v4(),
  "tenant_id" int4 NOT NULL,
  "knowledge_base_id" varchar(36) COLLATE "pg_catalog"."default" NOT NULL,
  "type" varchar(50) COLLATE "pg_catalog"."default" NOT NULL,
  "title" varchar(255) COLLATE "pg_catalog"."default" NOT NULL,
  "description" text COLLATE "pg_catalog"."default",
  "source" varchar(128) COLLATE "pg_catalog"."default" NOT NULL,
  "parse_status" varchar(50) COLLATE "pg_catalog"."default" NOT NULL DEFAULT 'unprocessed'::character varying,
  "enable_status" varchar(50) COLLATE "pg_catalog"."default" NOT NULL DEFAULT 'enabled'::character varying,
  "embedding_model_id" varchar(64) COLLATE "pg_catalog"."default",
  "file_name" varchar(255) COLLATE "pg_catalog"."default",
  "file_type" varchar(50) COLLATE "pg_catalog"."default",
  "file_size" int8,
  "file_path" text COLLATE "pg_catalog"."default",
  "file_hash" varchar(64) COLLATE "pg_catalog"."default",
  "storage_size" int8 NOT NULL DEFAULT 0,
  "metadata" jsonb,
  "created_at" timestamptz(6) DEFAULT CURRENT_TIMESTAMP,
  "updated_at" timestamptz(6) DEFAULT CURRENT_TIMESTAMP,
  "processed_at" timestamptz(6),
  "error_message" text COLLATE "pg_catalog"."default",
  "deleted_at" timestamptz(6),
  "tag_id" varchar(36) COLLATE "pg_catalog"."default",
  "summary_status" varchar(32) COLLATE "pg_catalog"."default" DEFAULT 'none'::character varying
)
;
ALTER TABLE "public"."knowledges" OWNER TO "postgres";

-- ----------------------------
-- Table structure for mcp_services
-- ----------------------------
DROP TABLE IF EXISTS "public"."mcp_services";
CREATE TABLE "public"."mcp_services" (
  "id" varchar(36) COLLATE "pg_catalog"."default" NOT NULL,
  "tenant_id" int4 NOT NULL,
  "name" varchar(255) COLLATE "pg_catalog"."default" NOT NULL,
  "description" text COLLATE "pg_catalog"."default",
  "enabled" bool DEFAULT true,
  "transport_type" varchar(50) COLLATE "pg_catalog"."default" NOT NULL,
  "url" varchar(512) COLLATE "pg_catalog"."default",
  "headers" jsonb,
  "auth_config" jsonb,
  "advanced_config" jsonb,
  "stdio_config" jsonb,
  "env_vars" jsonb,
  "created_at" timestamp(6) DEFAULT CURRENT_TIMESTAMP,
  "updated_at" timestamp(6) DEFAULT CURRENT_TIMESTAMP,
  "deleted_at" timestamp(6)
)
;
ALTER TABLE "public"."mcp_services" OWNER TO "postgres";
COMMENT ON TABLE "public"."mcp_services" IS 'MCP service configurations';

-- ----------------------------
-- Table structure for messages
-- ----------------------------
DROP TABLE IF EXISTS "public"."messages";
CREATE TABLE "public"."messages" (
  "id" varchar(36) COLLATE "pg_catalog"."default" NOT NULL DEFAULT uuid_generate_v4(),
  "request_id" varchar(36) COLLATE "pg_catalog"."default" NOT NULL,
  "session_id" varchar(36) COLLATE "pg_catalog"."default" NOT NULL,
  "role" varchar(50) COLLATE "pg_catalog"."default" NOT NULL,
  "content" text COLLATE "pg_catalog"."default" NOT NULL,
  "knowledge_references" jsonb NOT NULL DEFAULT '[]'::jsonb,
  "agent_steps" jsonb,
  "is_completed" bool NOT NULL DEFAULT false,
  "created_at" timestamptz(6) DEFAULT CURRENT_TIMESTAMP,
  "updated_at" timestamptz(6) DEFAULT CURRENT_TIMESTAMP,
  "deleted_at" timestamptz(6),
  "mentioned_items" jsonb DEFAULT '[]'::jsonb
)
;
ALTER TABLE "public"."messages" OWNER TO "postgres";
COMMENT ON COLUMN "public"."messages"."agent_steps" IS 'Agent execution steps (reasoning process and tool calls)';
COMMENT ON COLUMN "public"."messages"."mentioned_items" IS 'Stores @mentioned knowledge bases and files (id, name, type) when user sends a message';

-- ----------------------------
-- Table structure for models
-- ----------------------------
DROP TABLE IF EXISTS "public"."models";
CREATE TABLE "public"."models" (
  "id" varchar(64) COLLATE "pg_catalog"."default" NOT NULL DEFAULT uuid_generate_v4(),
  "tenant_id" int4 NOT NULL,
  "name" varchar(255) COLLATE "pg_catalog"."default" NOT NULL,
  "type" varchar(50) COLLATE "pg_catalog"."default" NOT NULL,
  "source" varchar(50) COLLATE "pg_catalog"."default" NOT NULL,
  "description" text COLLATE "pg_catalog"."default",
  "parameters" jsonb NOT NULL,
  "is_default" bool NOT NULL DEFAULT false,
  "status" varchar(50) COLLATE "pg_catalog"."default" NOT NULL DEFAULT 'active'::character varying,
  "created_at" timestamptz(6) DEFAULT CURRENT_TIMESTAMP,
  "updated_at" timestamptz(6) DEFAULT CURRENT_TIMESTAMP,
  "deleted_at" timestamptz(6),
  "is_builtin" bool NOT NULL DEFAULT false
)
;
ALTER TABLE "public"."models" OWNER TO "postgres";

-- ----------------------------
-- Table structure for schema_migrations
-- ----------------------------
DROP TABLE IF EXISTS "public"."schema_migrations";
CREATE TABLE "public"."schema_migrations" (
  "version" int8 NOT NULL,
  "dirty" bool NOT NULL
)
;
ALTER TABLE "public"."schema_migrations" OWNER TO "postgres";

-- ----------------------------
-- Table structure for session_items
-- ----------------------------
DROP TABLE IF EXISTS "public"."session_items";
CREATE TABLE "public"."session_items" (
  "id" varchar(36) COLLATE "pg_catalog"."default" NOT NULL DEFAULT uuid_generate_v4(),
  "session_id" varchar(36) COLLATE "pg_catalog"."default" NOT NULL,
  "type" varchar(20) COLLATE "pg_catalog"."default" NOT NULL,
  "message_id" varchar(36) COLLATE "pg_catalog"."default",
  "summary" text COLLATE "pg_catalog"."default",
  "sort_order" int4 NOT NULL,
  "token_count" int4 DEFAULT 0,
  "created_at" timestamptz(6) DEFAULT CURRENT_TIMESTAMP,
  "updated_at" timestamptz(6) DEFAULT CURRENT_TIMESTAMP
)
;
ALTER TABLE "public"."session_items" OWNER TO "postgres";
COMMENT ON COLUMN "public"."session_items"."type" IS 'Item type: "message" (references messages.id via message_id) or "summary" (stores summary text)';
COMMENT ON COLUMN "public"."session_items"."message_id" IS 'Foreign key reference to messages.id (used when type="message")';
COMMENT ON COLUMN "public"."session_items"."summary" IS 'Compressed summary text (used when type="summary")';
COMMENT ON COLUMN "public"."session_items"."sort_order" IS 'Order within session to maintain conversation sequence';
COMMENT ON COLUMN "public"."session_items"."token_count" IS 'Estimated token count for budget management';
COMMENT ON TABLE "public"."session_items" IS 'Session items for messages and summaries with unified sorting, supporting progressive compression';

-- ----------------------------
-- Table structure for sessions
-- ----------------------------
DROP TABLE IF EXISTS "public"."sessions";
CREATE TABLE "public"."sessions" (
  "id" varchar(36) COLLATE "pg_catalog"."default" NOT NULL DEFAULT uuid_generate_v4(),
  "tenant_id" int4 NOT NULL,
  "title" varchar(255) COLLATE "pg_catalog"."default",
  "description" text COLLATE "pg_catalog"."default",
  "knowledge_base_id" varchar(36) COLLATE "pg_catalog"."default",
  "max_rounds" int4 NOT NULL DEFAULT 5,
  "enable_rewrite" bool NOT NULL DEFAULT true,
  "fallback_strategy" varchar(255) COLLATE "pg_catalog"."default" NOT NULL DEFAULT 'fixed'::character varying,
  "fallback_response" text COLLATE "pg_catalog"."default" NOT NULL DEFAULT '很抱歉，我暂时无法回答这个问题。'::text,
  "keyword_threshold" float8 NOT NULL DEFAULT 0.5,
  "vector_threshold" float8 NOT NULL DEFAULT 0.5,
  "rerank_model_id" varchar(64) COLLATE "pg_catalog"."default",
  "embedding_top_k" int4 NOT NULL DEFAULT 10,
  "rerank_top_k" int4 NOT NULL DEFAULT 10,
  "rerank_threshold" float8 NOT NULL DEFAULT 0.65,
  "summary_model_id" varchar(64) COLLATE "pg_catalog"."default",
  "summary_parameters" jsonb NOT NULL DEFAULT '{}'::jsonb,
  "agent_config" jsonb,
  "context_config" jsonb,
  "created_at" timestamptz(6) DEFAULT CURRENT_TIMESTAMP,
  "updated_at" timestamptz(6) DEFAULT CURRENT_TIMESTAMP,
  "deleted_at" timestamptz(6),
  "agent_id" varchar(36) COLLATE "pg_catalog"."default"
)
;
ALTER TABLE "public"."sessions" OWNER TO "postgres";
COMMENT ON COLUMN "public"."sessions"."agent_config" IS 'Session-level agent configuration in JSON format';
COMMENT ON COLUMN "public"."sessions"."context_config" IS 'LLM context management configuration (separate from message storage)';

-- ----------------------------
-- Table structure for spatial_ref_sys
-- ----------------------------
DROP TABLE IF EXISTS "public"."spatial_ref_sys";
CREATE TABLE "public"."spatial_ref_sys" (
  "srid" int4 NOT NULL,
  "auth_name" varchar(256) COLLATE "pg_catalog"."default",
  "auth_srid" int4,
  "srtext" varchar(2048) COLLATE "pg_catalog"."default",
  "proj4text" varchar(2048) COLLATE "pg_catalog"."default"
)
;
ALTER TABLE "public"."spatial_ref_sys" OWNER TO "postgres";

-- ----------------------------
-- Table structure for tenants
-- ----------------------------
DROP TABLE IF EXISTS "public"."tenants";
CREATE TABLE "public"."tenants" (
  "id" int4 NOT NULL DEFAULT nextval('tenants_id_seq'::regclass),
  "name" varchar(255) COLLATE "pg_catalog"."default" NOT NULL,
  "description" text COLLATE "pg_catalog"."default",
  "api_key" varchar(64) COLLATE "pg_catalog"."default" NOT NULL,
  "retriever_engines" jsonb NOT NULL DEFAULT '[]'::jsonb,
  "status" varchar(50) COLLATE "pg_catalog"."default" DEFAULT 'active'::character varying,
  "business" varchar(255) COLLATE "pg_catalog"."default" NOT NULL,
  "storage_quota" int8 NOT NULL DEFAULT '10737418240'::bigint,
  "storage_used" int8 NOT NULL DEFAULT 0,
  "agent_config" jsonb,
  "created_at" timestamptz(6) DEFAULT CURRENT_TIMESTAMP,
  "updated_at" timestamptz(6) DEFAULT CURRENT_TIMESTAMP,
  "deleted_at" timestamptz(6),
  "context_config" jsonb,
  "conversation_config" jsonb,
  "web_search_config" jsonb
)
;
ALTER TABLE "public"."tenants" OWNER TO "postgres";
COMMENT ON COLUMN "public"."tenants"."agent_config" IS 'Tenant-level agent configuration in JSON format';
COMMENT ON COLUMN "public"."tenants"."context_config" IS 'Global Context configuration for this tenant (default for all sessions)';
COMMENT ON COLUMN "public"."tenants"."conversation_config" IS 'Global Conversation configuration for this tenant (default for normal mode sessions)';
COMMENT ON COLUMN "public"."tenants"."web_search_config" IS 'Web search configuration for the tenant';

-- ----------------------------
-- Table structure for users
-- ----------------------------
DROP TABLE IF EXISTS "public"."users";
CREATE TABLE "public"."users" (
  "id" varchar(36) COLLATE "pg_catalog"."default" NOT NULL DEFAULT uuid_generate_v4(),
  "username" varchar(100) COLLATE "pg_catalog"."default" NOT NULL,
  "email" varchar(255) COLLATE "pg_catalog"."default" NOT NULL,
  "password_hash" varchar(255) COLLATE "pg_catalog"."default" NOT NULL,
  "avatar" varchar(500) COLLATE "pg_catalog"."default",
  "tenant_id" int4,
  "is_active" bool NOT NULL DEFAULT true,
  "created_at" timestamptz(6) DEFAULT CURRENT_TIMESTAMP,
  "updated_at" timestamptz(6) DEFAULT CURRENT_TIMESTAMP,
  "deleted_at" timestamptz(6),
  "can_access_all_tenants" bool NOT NULL DEFAULT false
)
;
ALTER TABLE "public"."users" OWNER TO "postgres";
COMMENT ON COLUMN "public"."users"."id" IS 'Unique identifier of the user';
COMMENT ON COLUMN "public"."users"."username" IS 'Username of the user';
COMMENT ON COLUMN "public"."users"."email" IS 'Email address of the user';
COMMENT ON COLUMN "public"."users"."password_hash" IS 'Hashed password of the user';
COMMENT ON COLUMN "public"."users"."avatar" IS 'Avatar URL of the user';
COMMENT ON COLUMN "public"."users"."tenant_id" IS 'Tenant ID that the user belongs to';
COMMENT ON COLUMN "public"."users"."is_active" IS 'Whether the user is active';
COMMENT ON TABLE "public"."users" IS 'User accounts in the system';

-- ----------------------------
-- Function structure for _postgis_deprecate
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."_postgis_deprecate"("oldname" text, "newname" text, "version" text);
CREATE FUNCTION "public"."_postgis_deprecate"("oldname" text, "newname" text, "version" text)
  RETURNS "pg_catalog"."void" AS $BODY$
DECLARE
  curver_text text;
BEGIN
  --
  -- Raises a NOTICE if it was deprecated in this version,
  -- a WARNING if in a previous version (only up to minor version checked)
  --
	curver_text := '3.6.0';
	IF pg_catalog.split_part(curver_text,'.',1)::int > pg_catalog.split_part(version,'.',1)::int OR
	   ( pg_catalog.split_part(curver_text,'.',1) = pg_catalog.split_part(version,'.',1) AND
		 pg_catalog.split_part(curver_text,'.',2) != split_part(version,'.',2) )
	THEN
	  RAISE WARNING '% signature was deprecated in %. Please use %', oldname, version, newname;
	ELSE
	  RAISE DEBUG '% signature was deprecated in %. Please use %', oldname, version, newname;
	END IF;
END;
$BODY$
  LANGUAGE plpgsql IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."_postgis_deprecate"("oldname" text, "newname" text, "version" text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for _postgis_index_extent
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."_postgis_index_extent"("tbl" regclass, "col" text);
CREATE FUNCTION "public"."_postgis_index_extent"("tbl" regclass, "col" text)
  RETURNS "public"."box2d" AS '$libdir/postgis-3', '_postgis_gserialized_index_extent'
  LANGUAGE c STABLE STRICT
  COST 1;
ALTER FUNCTION "public"."_postgis_index_extent"("tbl" regclass, "col" text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for _postgis_join_selectivity
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."_postgis_join_selectivity"(regclass, text, regclass, text, text);
CREATE FUNCTION "public"."_postgis_join_selectivity"(regclass, text, regclass, text, text='2'::text)
  RETURNS "pg_catalog"."float8" AS '$libdir/postgis-3', '_postgis_gserialized_joinsel'
  LANGUAGE c VOLATILE STRICT
  COST 1;
ALTER FUNCTION "public"."_postgis_join_selectivity"(regclass, text, regclass, text, text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for _postgis_pgsql_version
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."_postgis_pgsql_version"();
CREATE FUNCTION "public"."_postgis_pgsql_version"()
  RETURNS "pg_catalog"."text" AS $BODY$
	SELECT CASE WHEN pg_catalog.split_part(s,'.',1)::integer > 9 THEN pg_catalog.split_part(s,'.',1) || '0'
	ELSE pg_catalog.split_part(s,'.', 1) || pg_catalog.split_part(s,'.', 2) END AS v
	FROM pg_catalog.substring(version(), E'PostgreSQL ([0-9\\.]+)') AS s;
$BODY$
  LANGUAGE sql STABLE
  COST 100;
ALTER FUNCTION "public"."_postgis_pgsql_version"() OWNER TO "postgres";

-- ----------------------------
-- Function structure for _postgis_scripts_pgsql_version
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."_postgis_scripts_pgsql_version"();
CREATE FUNCTION "public"."_postgis_scripts_pgsql_version"()
  RETURNS "pg_catalog"."text" AS $BODY$SELECT '170'::text AS version$BODY$
  LANGUAGE sql IMMUTABLE
  COST 100;
ALTER FUNCTION "public"."_postgis_scripts_pgsql_version"() OWNER TO "postgres";

-- ----------------------------
-- Function structure for _postgis_selectivity
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."_postgis_selectivity"("tbl" regclass, "att_name" text, "geom" "public"."geometry", "mode" text);
CREATE FUNCTION "public"."_postgis_selectivity"("tbl" regclass, "att_name" text, "geom" "public"."geometry", "mode" text='2'::text)
  RETURNS "pg_catalog"."float8" AS '$libdir/postgis-3', '_postgis_gserialized_sel'
  LANGUAGE c VOLATILE STRICT
  COST 1;
ALTER FUNCTION "public"."_postgis_selectivity"("tbl" regclass, "att_name" text, "geom" "public"."geometry", "mode" text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for _postgis_stats
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."_postgis_stats"("tbl" regclass, "att_name" text, text);
CREATE FUNCTION "public"."_postgis_stats"("tbl" regclass, "att_name" text, text='2'::text)
  RETURNS "pg_catalog"."text" AS '$libdir/postgis-3', '_postgis_gserialized_stats'
  LANGUAGE c VOLATILE STRICT
  COST 1;
ALTER FUNCTION "public"."_postgis_stats"("tbl" regclass, "att_name" text, text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for _st_3ddfullywithin
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."_st_3ddfullywithin"("geom1" "public"."geometry", "geom2" "public"."geometry", float8);
CREATE FUNCTION "public"."_st_3ddfullywithin"("geom1" "public"."geometry", "geom2" "public"."geometry", float8)
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'LWGEOM_dfullywithin3d'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."_st_3ddfullywithin"("geom1" "public"."geometry", "geom2" "public"."geometry", float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for _st_3ddwithin
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."_st_3ddwithin"("geom1" "public"."geometry", "geom2" "public"."geometry", float8);
CREATE FUNCTION "public"."_st_3ddwithin"("geom1" "public"."geometry", "geom2" "public"."geometry", float8)
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'LWGEOM_dwithin3d'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."_st_3ddwithin"("geom1" "public"."geometry", "geom2" "public"."geometry", float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for _st_3dintersects
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."_st_3dintersects"("geom1" "public"."geometry", "geom2" "public"."geometry");
CREATE FUNCTION "public"."_st_3dintersects"("geom1" "public"."geometry", "geom2" "public"."geometry")
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'ST_3DIntersects'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."_st_3dintersects"("geom1" "public"."geometry", "geom2" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for _st_asgml
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."_st_asgml"(int4, "public"."geometry", int4, int4, text, text);
CREATE FUNCTION "public"."_st_asgml"(int4, "public"."geometry", int4, int4, text, text)
  RETURNS "pg_catalog"."text" AS '$libdir/postgis-3', 'LWGEOM_asGML'
  LANGUAGE c IMMUTABLE
  COST 250;
ALTER FUNCTION "public"."_st_asgml"(int4, "public"."geometry", int4, int4, text, text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for _st_asx3d
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."_st_asx3d"(int4, "public"."geometry", int4, int4, text);
CREATE FUNCTION "public"."_st_asx3d"(int4, "public"."geometry", int4, int4, text)
  RETURNS "pg_catalog"."text" AS '$libdir/postgis-3', 'LWGEOM_asX3D'
  LANGUAGE c IMMUTABLE
  COST 250;
ALTER FUNCTION "public"."_st_asx3d"(int4, "public"."geometry", int4, int4, text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for _st_bestsrid
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."_st_bestsrid"("public"."geography");
CREATE FUNCTION "public"."_st_bestsrid"("public"."geography")
  RETURNS "pg_catalog"."int4" AS '$libdir/postgis-3', 'geography_bestsrid'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."_st_bestsrid"("public"."geography") OWNER TO "postgres";

-- ----------------------------
-- Function structure for _st_bestsrid
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."_st_bestsrid"("public"."geography", "public"."geography");
CREATE FUNCTION "public"."_st_bestsrid"("public"."geography", "public"."geography")
  RETURNS "pg_catalog"."int4" AS '$libdir/postgis-3', 'geography_bestsrid'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."_st_bestsrid"("public"."geography", "public"."geography") OWNER TO "postgres";

-- ----------------------------
-- Function structure for _st_contains
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."_st_contains"("geom1" "public"."geometry", "geom2" "public"."geometry");
CREATE FUNCTION "public"."_st_contains"("geom1" "public"."geometry", "geom2" "public"."geometry")
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'contains'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."_st_contains"("geom1" "public"."geometry", "geom2" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for _st_containsproperly
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."_st_containsproperly"("geom1" "public"."geometry", "geom2" "public"."geometry");
CREATE FUNCTION "public"."_st_containsproperly"("geom1" "public"."geometry", "geom2" "public"."geometry")
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'containsproperly'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."_st_containsproperly"("geom1" "public"."geometry", "geom2" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for _st_coveredby
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."_st_coveredby"("geog1" "public"."geography", "geog2" "public"."geography");
CREATE FUNCTION "public"."_st_coveredby"("geog1" "public"."geography", "geog2" "public"."geography")
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'geography_coveredby'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."_st_coveredby"("geog1" "public"."geography", "geog2" "public"."geography") OWNER TO "postgres";

-- ----------------------------
-- Function structure for _st_coveredby
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."_st_coveredby"("geom1" "public"."geometry", "geom2" "public"."geometry");
CREATE FUNCTION "public"."_st_coveredby"("geom1" "public"."geometry", "geom2" "public"."geometry")
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'coveredby'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."_st_coveredby"("geom1" "public"."geometry", "geom2" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for _st_covers
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."_st_covers"("geom1" "public"."geometry", "geom2" "public"."geometry");
CREATE FUNCTION "public"."_st_covers"("geom1" "public"."geometry", "geom2" "public"."geometry")
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'covers'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."_st_covers"("geom1" "public"."geometry", "geom2" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for _st_covers
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."_st_covers"("geog1" "public"."geography", "geog2" "public"."geography");
CREATE FUNCTION "public"."_st_covers"("geog1" "public"."geography", "geog2" "public"."geography")
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'geography_covers'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."_st_covers"("geog1" "public"."geography", "geog2" "public"."geography") OWNER TO "postgres";

-- ----------------------------
-- Function structure for _st_crosses
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."_st_crosses"("geom1" "public"."geometry", "geom2" "public"."geometry");
CREATE FUNCTION "public"."_st_crosses"("geom1" "public"."geometry", "geom2" "public"."geometry")
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'crosses'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."_st_crosses"("geom1" "public"."geometry", "geom2" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for _st_dfullywithin
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."_st_dfullywithin"("geom1" "public"."geometry", "geom2" "public"."geometry", float8);
CREATE FUNCTION "public"."_st_dfullywithin"("geom1" "public"."geometry", "geom2" "public"."geometry", float8)
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'LWGEOM_dfullywithin'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."_st_dfullywithin"("geom1" "public"."geometry", "geom2" "public"."geometry", float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for _st_distancetree
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."_st_distancetree"("public"."geography", "public"."geography", float8, bool);
CREATE FUNCTION "public"."_st_distancetree"("public"."geography", "public"."geography", float8, bool)
  RETURNS "pg_catalog"."float8" AS '$libdir/postgis-3', 'geography_distance_tree'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."_st_distancetree"("public"."geography", "public"."geography", float8, bool) OWNER TO "postgres";

-- ----------------------------
-- Function structure for _st_distancetree
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."_st_distancetree"("public"."geography", "public"."geography");
CREATE FUNCTION "public"."_st_distancetree"("public"."geography", "public"."geography")
  RETURNS "pg_catalog"."float8" AS $BODY$SELECT public._ST_DistanceTree($1, $2, 0.0, true)$BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 100;
ALTER FUNCTION "public"."_st_distancetree"("public"."geography", "public"."geography") OWNER TO "postgres";

-- ----------------------------
-- Function structure for _st_distanceuncached
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."_st_distanceuncached"("public"."geography", "public"."geography");
CREATE FUNCTION "public"."_st_distanceuncached"("public"."geography", "public"."geography")
  RETURNS "pg_catalog"."float8" AS $BODY$SELECT public._ST_DistanceUnCached($1, $2, 0.0, true)$BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 100;
ALTER FUNCTION "public"."_st_distanceuncached"("public"."geography", "public"."geography") OWNER TO "postgres";

-- ----------------------------
-- Function structure for _st_distanceuncached
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."_st_distanceuncached"("public"."geography", "public"."geography", float8, bool);
CREATE FUNCTION "public"."_st_distanceuncached"("public"."geography", "public"."geography", float8, bool)
  RETURNS "pg_catalog"."float8" AS '$libdir/postgis-3', 'geography_distance_uncached'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."_st_distanceuncached"("public"."geography", "public"."geography", float8, bool) OWNER TO "postgres";

-- ----------------------------
-- Function structure for _st_distanceuncached
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."_st_distanceuncached"("public"."geography", "public"."geography", bool);
CREATE FUNCTION "public"."_st_distanceuncached"("public"."geography", "public"."geography", bool)
  RETURNS "pg_catalog"."float8" AS $BODY$SELECT public._ST_DistanceUnCached($1, $2, 0.0, $3)$BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 100;
ALTER FUNCTION "public"."_st_distanceuncached"("public"."geography", "public"."geography", bool) OWNER TO "postgres";

-- ----------------------------
-- Function structure for _st_dwithin
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."_st_dwithin"("geom1" "public"."geometry", "geom2" "public"."geometry", float8);
CREATE FUNCTION "public"."_st_dwithin"("geom1" "public"."geometry", "geom2" "public"."geometry", float8)
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'LWGEOM_dwithin'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."_st_dwithin"("geom1" "public"."geometry", "geom2" "public"."geometry", float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for _st_dwithin
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."_st_dwithin"("geog1" "public"."geography", "geog2" "public"."geography", "tolerance" float8, "use_spheroid" bool);
CREATE FUNCTION "public"."_st_dwithin"("geog1" "public"."geography", "geog2" "public"."geography", "tolerance" float8, "use_spheroid" bool=true)
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'geography_dwithin'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."_st_dwithin"("geog1" "public"."geography", "geog2" "public"."geography", "tolerance" float8, "use_spheroid" bool) OWNER TO "postgres";

-- ----------------------------
-- Function structure for _st_dwithinuncached
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."_st_dwithinuncached"("public"."geography", "public"."geography", float8);
CREATE FUNCTION "public"."_st_dwithinuncached"("public"."geography", "public"."geography", float8)
  RETURNS "pg_catalog"."bool" AS $BODY$SELECT $1 OPERATOR(public.&&) public._ST_Expand($2,$3) AND $2 OPERATOR(public.&&) public._ST_Expand($1,$3) AND public._ST_DWithinUnCached($1, $2, $3, true)$BODY$
  LANGUAGE sql IMMUTABLE
  COST 100;
ALTER FUNCTION "public"."_st_dwithinuncached"("public"."geography", "public"."geography", float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for _st_dwithinuncached
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."_st_dwithinuncached"("public"."geography", "public"."geography", float8, bool);
CREATE FUNCTION "public"."_st_dwithinuncached"("public"."geography", "public"."geography", float8, bool)
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'geography_dwithin_uncached'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."_st_dwithinuncached"("public"."geography", "public"."geography", float8, bool) OWNER TO "postgres";

-- ----------------------------
-- Function structure for _st_equals
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."_st_equals"("geom1" "public"."geometry", "geom2" "public"."geometry");
CREATE FUNCTION "public"."_st_equals"("geom1" "public"."geometry", "geom2" "public"."geometry")
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'ST_Equals'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."_st_equals"("geom1" "public"."geometry", "geom2" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for _st_expand
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."_st_expand"("public"."geography", float8);
CREATE FUNCTION "public"."_st_expand"("public"."geography", float8)
  RETURNS "public"."geography" AS '$libdir/postgis-3', 'geography_expand'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."_st_expand"("public"."geography", float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for _st_geomfromgml
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."_st_geomfromgml"(text, int4);
CREATE FUNCTION "public"."_st_geomfromgml"(text, int4)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'geom_from_gml'
  LANGUAGE c IMMUTABLE
  COST 250;
ALTER FUNCTION "public"."_st_geomfromgml"(text, int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for _st_intersects
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."_st_intersects"("geom1" "public"."geometry", "geom2" "public"."geometry");
CREATE FUNCTION "public"."_st_intersects"("geom1" "public"."geometry", "geom2" "public"."geometry")
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'ST_Intersects'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."_st_intersects"("geom1" "public"."geometry", "geom2" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for _st_linecrossingdirection
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."_st_linecrossingdirection"("line1" "public"."geometry", "line2" "public"."geometry");
CREATE FUNCTION "public"."_st_linecrossingdirection"("line1" "public"."geometry", "line2" "public"."geometry")
  RETURNS "pg_catalog"."int4" AS '$libdir/postgis-3', 'ST_LineCrossingDirection'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."_st_linecrossingdirection"("line1" "public"."geometry", "line2" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for _st_longestline
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."_st_longestline"("geom1" "public"."geometry", "geom2" "public"."geometry");
CREATE FUNCTION "public"."_st_longestline"("geom1" "public"."geometry", "geom2" "public"."geometry")
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'LWGEOM_longestline2d'
  LANGUAGE c IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."_st_longestline"("geom1" "public"."geometry", "geom2" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for _st_maxdistance
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."_st_maxdistance"("geom1" "public"."geometry", "geom2" "public"."geometry");
CREATE FUNCTION "public"."_st_maxdistance"("geom1" "public"."geometry", "geom2" "public"."geometry")
  RETURNS "pg_catalog"."float8" AS '$libdir/postgis-3', 'LWGEOM_maxdistance2d_linestring'
  LANGUAGE c IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."_st_maxdistance"("geom1" "public"."geometry", "geom2" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for _st_orderingequals
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."_st_orderingequals"("geom1" "public"."geometry", "geom2" "public"."geometry");
CREATE FUNCTION "public"."_st_orderingequals"("geom1" "public"."geometry", "geom2" "public"."geometry")
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'LWGEOM_same'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."_st_orderingequals"("geom1" "public"."geometry", "geom2" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for _st_overlaps
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."_st_overlaps"("geom1" "public"."geometry", "geom2" "public"."geometry");
CREATE FUNCTION "public"."_st_overlaps"("geom1" "public"."geometry", "geom2" "public"."geometry")
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'overlaps'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."_st_overlaps"("geom1" "public"."geometry", "geom2" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for _st_pointoutside
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."_st_pointoutside"("public"."geography");
CREATE FUNCTION "public"."_st_pointoutside"("public"."geography")
  RETURNS "public"."geography" AS '$libdir/postgis-3', 'geography_point_outside'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."_st_pointoutside"("public"."geography") OWNER TO "postgres";

-- ----------------------------
-- Function structure for _st_sortablehash
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."_st_sortablehash"("geom" "public"."geometry");
CREATE FUNCTION "public"."_st_sortablehash"("geom" "public"."geometry")
  RETURNS "pg_catalog"."int8" AS '$libdir/postgis-3', '_ST_SortableHash'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."_st_sortablehash"("geom" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for _st_touches
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."_st_touches"("geom1" "public"."geometry", "geom2" "public"."geometry");
CREATE FUNCTION "public"."_st_touches"("geom1" "public"."geometry", "geom2" "public"."geometry")
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'touches'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."_st_touches"("geom1" "public"."geometry", "geom2" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for _st_voronoi
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."_st_voronoi"("g1" "public"."geometry", "clip" "public"."geometry", "tolerance" float8, "return_polygons" bool);
CREATE FUNCTION "public"."_st_voronoi"("g1" "public"."geometry", "clip" "public"."geometry"=NULL::geometry, "tolerance" float8=0.0, "return_polygons" bool=true)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'ST_Voronoi'
  LANGUAGE c IMMUTABLE
  COST 5000;
ALTER FUNCTION "public"."_st_voronoi"("g1" "public"."geometry", "clip" "public"."geometry", "tolerance" float8, "return_polygons" bool) OWNER TO "postgres";

-- ----------------------------
-- Function structure for _st_within
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."_st_within"("geom1" "public"."geometry", "geom2" "public"."geometry");
CREATE FUNCTION "public"."_st_within"("geom1" "public"."geometry", "geom2" "public"."geometry")
  RETURNS "pg_catalog"."bool" AS $BODY$SELECT public._ST_Contains($2,$1)$BODY$
  LANGUAGE sql IMMUTABLE
  COST 100;
ALTER FUNCTION "public"."_st_within"("geom1" "public"."geometry", "geom2" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for addgeometrycolumn
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."addgeometrycolumn"("schema_name" varchar, "table_name" varchar, "column_name" varchar, "new_srid" int4, "new_type" varchar, "new_dim" int4, "use_typmod" bool);
CREATE FUNCTION "public"."addgeometrycolumn"("schema_name" varchar, "table_name" varchar, "column_name" varchar, "new_srid" int4, "new_type" varchar, "new_dim" int4, "use_typmod" bool=true)
  RETURNS "pg_catalog"."text" AS $BODY$
DECLARE
	ret  text;
BEGIN
	SELECT public.AddGeometryColumn('',$1,$2,$3,$4,$5,$6,$7) into ret;
	RETURN ret;
END;
$BODY$
  LANGUAGE plpgsql STABLE STRICT
  COST 100;
ALTER FUNCTION "public"."addgeometrycolumn"("schema_name" varchar, "table_name" varchar, "column_name" varchar, "new_srid" int4, "new_type" varchar, "new_dim" int4, "use_typmod" bool) OWNER TO "postgres";

-- ----------------------------
-- Function structure for addgeometrycolumn
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."addgeometrycolumn"("table_name" varchar, "column_name" varchar, "new_srid" int4, "new_type" varchar, "new_dim" int4, "use_typmod" bool);
CREATE FUNCTION "public"."addgeometrycolumn"("table_name" varchar, "column_name" varchar, "new_srid" int4, "new_type" varchar, "new_dim" int4, "use_typmod" bool=true)
  RETURNS "pg_catalog"."text" AS $BODY$
DECLARE
	ret  text;
BEGIN
	SELECT public.AddGeometryColumn('','',$1,$2,$3,$4,$5, $6) into ret;
	RETURN ret;
END;
$BODY$
  LANGUAGE plpgsql VOLATILE STRICT
  COST 100;
ALTER FUNCTION "public"."addgeometrycolumn"("table_name" varchar, "column_name" varchar, "new_srid" int4, "new_type" varchar, "new_dim" int4, "use_typmod" bool) OWNER TO "postgres";

-- ----------------------------
-- Function structure for addgeometrycolumn
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."addgeometrycolumn"("catalog_name" varchar, "schema_name" varchar, "table_name" varchar, "column_name" varchar, "new_srid_in" int4, "new_type" varchar, "new_dim" int4, "use_typmod" bool);
CREATE FUNCTION "public"."addgeometrycolumn"("catalog_name" varchar, "schema_name" varchar, "table_name" varchar, "column_name" varchar, "new_srid_in" int4, "new_type" varchar, "new_dim" int4, "use_typmod" bool=true)
  RETURNS "pg_catalog"."text" AS $BODY$
DECLARE
	rec RECORD;
	sr varchar;
	real_schema name;
	sql text;
	new_srid integer;

BEGIN

	-- Verify geometry type
	IF (postgis_type_name(new_type,new_dim) IS NULL )
	THEN
		RAISE EXCEPTION 'Invalid type name "%(%)" - valid ones are:
	POINT, MULTIPOINT,
	LINESTRING, MULTILINESTRING,
	POLYGON, MULTIPOLYGON,
	CIRCULARSTRING, COMPOUNDCURVE, MULTICURVE,
	CURVEPOLYGON, MULTISURFACE,
	GEOMETRY, GEOMETRYCOLLECTION,
	POINTM, MULTIPOINTM,
	LINESTRINGM, MULTILINESTRINGM,
	POLYGONM, MULTIPOLYGONM,
	CIRCULARSTRINGM, COMPOUNDCURVEM, MULTICURVEM
	CURVEPOLYGONM, MULTISURFACEM, TRIANGLE, TRIANGLEM,
	POLYHEDRALSURFACE, POLYHEDRALSURFACEM, TIN, TINM
	or GEOMETRYCOLLECTIONM', new_type, new_dim;
		RETURN 'fail';
	END IF;

	-- Verify dimension
	IF ( (new_dim >4) OR (new_dim <2) ) THEN
		RAISE EXCEPTION 'invalid dimension';
		RETURN 'fail';
	END IF;

	IF ( (new_type LIKE '%M') AND (new_dim!=3) ) THEN
		RAISE EXCEPTION 'TypeM needs 3 dimensions';
		RETURN 'fail';
	END IF;

	-- Verify SRID
	IF ( new_srid_in > 0 ) THEN
		IF new_srid_in > 998999 THEN
			RAISE EXCEPTION 'AddGeometryColumn() - SRID must be <= %', 998999;
		END IF;
		new_srid := new_srid_in;
		SELECT SRID INTO sr FROM public.spatial_ref_sys WHERE SRID = new_srid;
		IF NOT FOUND THEN
			RAISE EXCEPTION 'AddGeometryColumn() - invalid SRID';
			RETURN 'fail';
		END IF;
	ELSE
		new_srid := public.ST_SRID('POINT EMPTY'::public.geometry);
		IF ( new_srid_in != new_srid ) THEN
			RAISE NOTICE 'SRID value % converted to the officially unknown SRID value %', new_srid_in, new_srid;
		END IF;
	END IF;

	-- Verify schema
	IF ( schema_name IS NOT NULL AND schema_name != '' ) THEN
		sql := 'SELECT nspname FROM pg_namespace ' ||
			'WHERE text(nspname) = ' || quote_literal(schema_name) ||
			'LIMIT 1';
		RAISE DEBUG '%', sql;
		EXECUTE sql INTO real_schema;

		IF ( real_schema IS NULL ) THEN
			RAISE EXCEPTION 'Schema % is not a valid schemaname', quote_literal(schema_name);
			RETURN 'fail';
		END IF;
	END IF;

	IF ( real_schema IS NULL ) THEN
		RAISE DEBUG 'Detecting schema';
		sql := 'SELECT n.nspname AS schemaname ' ||
			'FROM pg_catalog.pg_class c ' ||
			  'JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace ' ||
			'WHERE c.relkind = ' || quote_literal('r') ||
			' AND n.nspname NOT IN (' || quote_literal('pg_catalog') || ', ' || quote_literal('pg_toast') || ')' ||
			' AND pg_catalog.pg_table_is_visible(c.oid)' ||
			' AND c.relname = ' || quote_literal(table_name);
		RAISE DEBUG '%', sql;
		EXECUTE sql INTO real_schema;

		IF ( real_schema IS NULL ) THEN
			RAISE EXCEPTION 'Table % does not occur in the search_path', quote_literal(table_name);
			RETURN 'fail';
		END IF;
	END IF;

	-- Add geometry column to table
	IF use_typmod THEN
		 sql := 'ALTER TABLE ' ||
			quote_ident(real_schema) || '.' || quote_ident(table_name)
			|| ' ADD COLUMN ' || quote_ident(column_name) ||
			' geometry(' || public.postgis_type_name(new_type, new_dim) || ', ' || new_srid::text || ')';
		RAISE DEBUG '%', sql;
	ELSE
		sql := 'ALTER TABLE ' ||
			quote_ident(real_schema) || '.' || quote_ident(table_name)
			|| ' ADD COLUMN ' || quote_ident(column_name) ||
			' geometry ';
		RAISE DEBUG '%', sql;
	END IF;
	EXECUTE sql;

	IF NOT use_typmod THEN
		-- Add table CHECKs
		sql := 'ALTER TABLE ' ||
			quote_ident(real_schema) || '.' || quote_ident(table_name)
			|| ' ADD CONSTRAINT '
			|| quote_ident('enforce_srid_' || column_name)
			|| ' CHECK (st_srid(' || quote_ident(column_name) ||
			') = ' || new_srid::text || ')' ;
		RAISE DEBUG '%', sql;
		EXECUTE sql;

		sql := 'ALTER TABLE ' ||
			quote_ident(real_schema) || '.' || quote_ident(table_name)
			|| ' ADD CONSTRAINT '
			|| quote_ident('enforce_dims_' || column_name)
			|| ' CHECK (st_ndims(' || quote_ident(column_name) ||
			') = ' || new_dim::text || ')' ;
		RAISE DEBUG '%', sql;
		EXECUTE sql;

		IF ( NOT (new_type = 'GEOMETRY')) THEN
			sql := 'ALTER TABLE ' ||
				quote_ident(real_schema) || '.' || quote_ident(table_name) || ' ADD CONSTRAINT ' ||
				quote_ident('enforce_geotype_' || column_name) ||
				' CHECK (GeometryType(' ||
				quote_ident(column_name) || ')=' ||
				quote_literal(new_type) || ' OR (' ||
				quote_ident(column_name) || ') is null)';
			RAISE DEBUG '%', sql;
			EXECUTE sql;
		END IF;
	END IF;

	RETURN
		real_schema || '.' ||
		table_name || '.' || column_name ||
		' SRID:' || new_srid::text ||
		' TYPE:' || new_type ||
		' DIMS:' || new_dim::text || ' ';
END;
$BODY$
  LANGUAGE plpgsql VOLATILE STRICT
  COST 100;
ALTER FUNCTION "public"."addgeometrycolumn"("catalog_name" varchar, "schema_name" varchar, "table_name" varchar, "column_name" varchar, "new_srid_in" int4, "new_type" varchar, "new_dim" int4, "use_typmod" bool) OWNER TO "postgres";

-- ----------------------------
-- Function structure for array_to_halfvec
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."array_to_halfvec"(_numeric, int4, bool);
CREATE FUNCTION "public"."array_to_halfvec"(_numeric, int4, bool)
  RETURNS "public"."halfvec" AS '$libdir/vector', 'array_to_halfvec'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."array_to_halfvec"(_numeric, int4, bool) OWNER TO "postgres";

-- ----------------------------
-- Function structure for array_to_halfvec
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."array_to_halfvec"(_int4, int4, bool);
CREATE FUNCTION "public"."array_to_halfvec"(_int4, int4, bool)
  RETURNS "public"."halfvec" AS '$libdir/vector', 'array_to_halfvec'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."array_to_halfvec"(_int4, int4, bool) OWNER TO "postgres";

-- ----------------------------
-- Function structure for array_to_halfvec
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."array_to_halfvec"(_float4, int4, bool);
CREATE FUNCTION "public"."array_to_halfvec"(_float4, int4, bool)
  RETURNS "public"."halfvec" AS '$libdir/vector', 'array_to_halfvec'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."array_to_halfvec"(_float4, int4, bool) OWNER TO "postgres";

-- ----------------------------
-- Function structure for array_to_halfvec
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."array_to_halfvec"(_float8, int4, bool);
CREATE FUNCTION "public"."array_to_halfvec"(_float8, int4, bool)
  RETURNS "public"."halfvec" AS '$libdir/vector', 'array_to_halfvec'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."array_to_halfvec"(_float8, int4, bool) OWNER TO "postgres";

-- ----------------------------
-- Function structure for array_to_sparsevec
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."array_to_sparsevec"(_int4, int4, bool);
CREATE FUNCTION "public"."array_to_sparsevec"(_int4, int4, bool)
  RETURNS "public"."sparsevec" AS '$libdir/vector', 'array_to_sparsevec'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."array_to_sparsevec"(_int4, int4, bool) OWNER TO "postgres";

-- ----------------------------
-- Function structure for array_to_sparsevec
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."array_to_sparsevec"(_float4, int4, bool);
CREATE FUNCTION "public"."array_to_sparsevec"(_float4, int4, bool)
  RETURNS "public"."sparsevec" AS '$libdir/vector', 'array_to_sparsevec'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."array_to_sparsevec"(_float4, int4, bool) OWNER TO "postgres";

-- ----------------------------
-- Function structure for array_to_sparsevec
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."array_to_sparsevec"(_float8, int4, bool);
CREATE FUNCTION "public"."array_to_sparsevec"(_float8, int4, bool)
  RETURNS "public"."sparsevec" AS '$libdir/vector', 'array_to_sparsevec'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."array_to_sparsevec"(_float8, int4, bool) OWNER TO "postgres";

-- ----------------------------
-- Function structure for array_to_sparsevec
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."array_to_sparsevec"(_numeric, int4, bool);
CREATE FUNCTION "public"."array_to_sparsevec"(_numeric, int4, bool)
  RETURNS "public"."sparsevec" AS '$libdir/vector', 'array_to_sparsevec'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."array_to_sparsevec"(_numeric, int4, bool) OWNER TO "postgres";

-- ----------------------------
-- Function structure for array_to_vector
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."array_to_vector"(_float8, int4, bool);
CREATE FUNCTION "public"."array_to_vector"(_float8, int4, bool)
  RETURNS "public"."vector" AS '$libdir/vector', 'array_to_vector'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."array_to_vector"(_float8, int4, bool) OWNER TO "postgres";

-- ----------------------------
-- Function structure for array_to_vector
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."array_to_vector"(_int4, int4, bool);
CREATE FUNCTION "public"."array_to_vector"(_int4, int4, bool)
  RETURNS "public"."vector" AS '$libdir/vector', 'array_to_vector'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."array_to_vector"(_int4, int4, bool) OWNER TO "postgres";

-- ----------------------------
-- Function structure for array_to_vector
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."array_to_vector"(_float4, int4, bool);
CREATE FUNCTION "public"."array_to_vector"(_float4, int4, bool)
  RETURNS "public"."vector" AS '$libdir/vector', 'array_to_vector'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."array_to_vector"(_float4, int4, bool) OWNER TO "postgres";

-- ----------------------------
-- Function structure for array_to_vector
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."array_to_vector"(_numeric, int4, bool);
CREATE FUNCTION "public"."array_to_vector"(_numeric, int4, bool)
  RETURNS "public"."vector" AS '$libdir/vector', 'array_to_vector'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."array_to_vector"(_numeric, int4, bool) OWNER TO "postgres";

-- ----------------------------
-- Function structure for binary_quantize
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."binary_quantize"("public"."halfvec");
CREATE FUNCTION "public"."binary_quantize"("public"."halfvec")
  RETURNS "pg_catalog"."bit" AS '$libdir/vector', 'halfvec_binary_quantize'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."binary_quantize"("public"."halfvec") OWNER TO "postgres";

-- ----------------------------
-- Function structure for binary_quantize
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."binary_quantize"("public"."vector");
CREATE FUNCTION "public"."binary_quantize"("public"."vector")
  RETURNS "pg_catalog"."bit" AS '$libdir/vector', 'binary_quantize'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."binary_quantize"("public"."vector") OWNER TO "postgres";

-- ----------------------------
-- Function structure for box
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."box"("public"."geometry");
CREATE FUNCTION "public"."box"("public"."geometry")
  RETURNS "pg_catalog"."box" AS '$libdir/postgis-3', 'LWGEOM_to_BOX'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."box"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for box
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."box"("public"."box3d");
CREATE FUNCTION "public"."box"("public"."box3d")
  RETURNS "pg_catalog"."box" AS '$libdir/postgis-3', 'BOX3D_to_BOX'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."box"("public"."box3d") OWNER TO "postgres";

-- ----------------------------
-- Function structure for box2d
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."box2d"("public"."box3d");
CREATE FUNCTION "public"."box2d"("public"."box3d")
  RETURNS "public"."box2d" AS '$libdir/postgis-3', 'BOX3D_to_BOX2D'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."box2d"("public"."box3d") OWNER TO "postgres";

-- ----------------------------
-- Function structure for box2d
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."box2d"("public"."geometry");
CREATE FUNCTION "public"."box2d"("public"."geometry")
  RETURNS "public"."box2d" AS '$libdir/postgis-3', 'LWGEOM_to_BOX2D'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."box2d"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for box2d_in
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."box2d_in"(cstring);
CREATE FUNCTION "public"."box2d_in"(cstring)
  RETURNS "public"."box2d" AS '$libdir/postgis-3', 'BOX2D_in'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."box2d_in"(cstring) OWNER TO "postgres";

-- ----------------------------
-- Function structure for box2d_out
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."box2d_out"("public"."box2d");
CREATE FUNCTION "public"."box2d_out"("public"."box2d")
  RETURNS "pg_catalog"."cstring" AS '$libdir/postgis-3', 'BOX2D_out'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."box2d_out"("public"."box2d") OWNER TO "postgres";

-- ----------------------------
-- Function structure for box2df_in
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."box2df_in"(cstring);
CREATE FUNCTION "public"."box2df_in"(cstring)
  RETURNS "public"."box2df" AS '$libdir/postgis-3', 'box2df_in'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."box2df_in"(cstring) OWNER TO "postgres";

-- ----------------------------
-- Function structure for box2df_out
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."box2df_out"("public"."box2df");
CREATE FUNCTION "public"."box2df_out"("public"."box2df")
  RETURNS "pg_catalog"."cstring" AS '$libdir/postgis-3', 'box2df_out'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."box2df_out"("public"."box2df") OWNER TO "postgres";

-- ----------------------------
-- Function structure for box3d
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."box3d"("public"."geometry");
CREATE FUNCTION "public"."box3d"("public"."geometry")
  RETURNS "public"."box3d" AS '$libdir/postgis-3', 'LWGEOM_to_BOX3D'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."box3d"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for box3d
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."box3d"("public"."box2d");
CREATE FUNCTION "public"."box3d"("public"."box2d")
  RETURNS "public"."box3d" AS '$libdir/postgis-3', 'BOX2D_to_BOX3D'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."box3d"("public"."box2d") OWNER TO "postgres";

-- ----------------------------
-- Function structure for box3d_in
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."box3d_in"(cstring);
CREATE FUNCTION "public"."box3d_in"(cstring)
  RETURNS "public"."box3d" AS '$libdir/postgis-3', 'BOX3D_in'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."box3d_in"(cstring) OWNER TO "postgres";

-- ----------------------------
-- Function structure for box3d_out
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."box3d_out"("public"."box3d");
CREATE FUNCTION "public"."box3d_out"("public"."box3d")
  RETURNS "pg_catalog"."cstring" AS '$libdir/postgis-3', 'BOX3D_out'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."box3d_out"("public"."box3d") OWNER TO "postgres";

-- ----------------------------
-- Function structure for box3dtobox
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."box3dtobox"("public"."box3d");
CREATE FUNCTION "public"."box3dtobox"("public"."box3d")
  RETURNS "pg_catalog"."box" AS '$libdir/postgis-3', 'BOX3D_to_BOX'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."box3dtobox"("public"."box3d") OWNER TO "postgres";

-- ----------------------------
-- Function structure for bytea
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."bytea"("public"."geography");
CREATE FUNCTION "public"."bytea"("public"."geography")
  RETURNS "pg_catalog"."bytea" AS '$libdir/postgis-3', 'LWGEOM_to_bytea'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."bytea"("public"."geography") OWNER TO "postgres";

-- ----------------------------
-- Function structure for bytea
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."bytea"("public"."geometry");
CREATE FUNCTION "public"."bytea"("public"."geometry")
  RETURNS "pg_catalog"."bytea" AS '$libdir/postgis-3', 'LWGEOM_to_bytea'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."bytea"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for contains_2d
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."contains_2d"("public"."box2df", "public"."box2df");
CREATE FUNCTION "public"."contains_2d"("public"."box2df", "public"."box2df")
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'gserialized_contains_box2df_box2df_2d'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."contains_2d"("public"."box2df", "public"."box2df") OWNER TO "postgres";

-- ----------------------------
-- Function structure for contains_2d
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."contains_2d"("public"."box2df", "public"."geometry");
CREATE FUNCTION "public"."contains_2d"("public"."box2df", "public"."geometry")
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'gserialized_contains_box2df_geom_2d'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."contains_2d"("public"."box2df", "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for contains_2d
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."contains_2d"("public"."geometry", "public"."box2df");
CREATE FUNCTION "public"."contains_2d"("public"."geometry", "public"."box2df")
  RETURNS "pg_catalog"."bool" AS $BODY$SELECT $2 OPERATOR(public.@) $1;$BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."contains_2d"("public"."geometry", "public"."box2df") OWNER TO "postgres";

-- ----------------------------
-- Function structure for cosine_distance
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."cosine_distance"("public"."halfvec", "public"."halfvec");
CREATE FUNCTION "public"."cosine_distance"("public"."halfvec", "public"."halfvec")
  RETURNS "pg_catalog"."float8" AS '$libdir/vector', 'halfvec_cosine_distance'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."cosine_distance"("public"."halfvec", "public"."halfvec") OWNER TO "postgres";

-- ----------------------------
-- Function structure for cosine_distance
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."cosine_distance"("public"."sparsevec", "public"."sparsevec");
CREATE FUNCTION "public"."cosine_distance"("public"."sparsevec", "public"."sparsevec")
  RETURNS "pg_catalog"."float8" AS '$libdir/vector', 'sparsevec_cosine_distance'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."cosine_distance"("public"."sparsevec", "public"."sparsevec") OWNER TO "postgres";

-- ----------------------------
-- Function structure for cosine_distance
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."cosine_distance"("public"."vector", "public"."vector");
CREATE FUNCTION "public"."cosine_distance"("public"."vector", "public"."vector")
  RETURNS "pg_catalog"."float8" AS '$libdir/vector', 'cosine_distance'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."cosine_distance"("public"."vector", "public"."vector") OWNER TO "postgres";

-- ----------------------------
-- Function structure for daitch_mokotoff
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."daitch_mokotoff"(text);
CREATE FUNCTION "public"."daitch_mokotoff"(text)
  RETURNS "pg_catalog"."_text" AS '$libdir/fuzzystrmatch', 'daitch_mokotoff'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."daitch_mokotoff"(text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for difference
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."difference"(text, text);
CREATE FUNCTION "public"."difference"(text, text)
  RETURNS "pg_catalog"."int4" AS '$libdir/fuzzystrmatch', 'difference'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."difference"(text, text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for dmetaphone
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."dmetaphone"(text);
CREATE FUNCTION "public"."dmetaphone"(text)
  RETURNS "pg_catalog"."text" AS '$libdir/fuzzystrmatch', 'dmetaphone'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."dmetaphone"(text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for dmetaphone_alt
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."dmetaphone_alt"(text);
CREATE FUNCTION "public"."dmetaphone_alt"(text)
  RETURNS "pg_catalog"."text" AS '$libdir/fuzzystrmatch', 'dmetaphone_alt'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."dmetaphone_alt"(text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for dropgeometrycolumn
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."dropgeometrycolumn"("catalog_name" varchar, "schema_name" varchar, "table_name" varchar, "column_name" varchar);
CREATE FUNCTION "public"."dropgeometrycolumn"("catalog_name" varchar, "schema_name" varchar, "table_name" varchar, "column_name" varchar)
  RETURNS "pg_catalog"."text" AS $BODY$
DECLARE
	myrec RECORD;
	okay boolean;
	real_schema name;

BEGIN

	-- Find, check or fix schema_name
	IF ( schema_name != '' ) THEN
		okay = false;

		FOR myrec IN SELECT nspname FROM pg_namespace WHERE text(nspname) = schema_name LOOP
			okay := true;
		END LOOP;

		IF ( okay <>  true ) THEN
			RAISE NOTICE 'Invalid schema name - using current_schema()';
			SELECT current_schema() into real_schema;
		ELSE
			real_schema = schema_name;
		END IF;
	ELSE
		SELECT current_schema() into real_schema;
	END IF;

	-- Find out if the column is in the geometry_columns table
	okay = false;
	FOR myrec IN SELECT * from public.geometry_columns where f_table_schema = text(real_schema) and f_table_name = table_name and f_geometry_column = column_name LOOP
		okay := true;
	END LOOP;
	IF (okay <> true) THEN
		RAISE EXCEPTION 'column not found in geometry_columns table';
		RETURN false;
	END IF;

	-- Remove table column
	EXECUTE 'ALTER TABLE ' || quote_ident(real_schema) || '.' ||
		quote_ident(table_name) || ' DROP COLUMN ' ||
		quote_ident(column_name);

	RETURN real_schema || '.' || table_name || '.' || column_name ||' effectively removed.';

END;
$BODY$
  LANGUAGE plpgsql VOLATILE STRICT
  COST 100;
ALTER FUNCTION "public"."dropgeometrycolumn"("catalog_name" varchar, "schema_name" varchar, "table_name" varchar, "column_name" varchar) OWNER TO "postgres";

-- ----------------------------
-- Function structure for dropgeometrycolumn
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."dropgeometrycolumn"("table_name" varchar, "column_name" varchar);
CREATE FUNCTION "public"."dropgeometrycolumn"("table_name" varchar, "column_name" varchar)
  RETURNS "pg_catalog"."text" AS $BODY$
DECLARE
	ret text;
BEGIN
	SELECT public.DropGeometryColumn('','',$1,$2) into ret;
	RETURN ret;
END;
$BODY$
  LANGUAGE plpgsql VOLATILE STRICT
  COST 100;
ALTER FUNCTION "public"."dropgeometrycolumn"("table_name" varchar, "column_name" varchar) OWNER TO "postgres";

-- ----------------------------
-- Function structure for dropgeometrycolumn
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."dropgeometrycolumn"("schema_name" varchar, "table_name" varchar, "column_name" varchar);
CREATE FUNCTION "public"."dropgeometrycolumn"("schema_name" varchar, "table_name" varchar, "column_name" varchar)
  RETURNS "pg_catalog"."text" AS $BODY$
DECLARE
	ret text;
BEGIN
	SELECT public.DropGeometryColumn('',$1,$2,$3) into ret;
	RETURN ret;
END;
$BODY$
  LANGUAGE plpgsql VOLATILE STRICT
  COST 100;
ALTER FUNCTION "public"."dropgeometrycolumn"("schema_name" varchar, "table_name" varchar, "column_name" varchar) OWNER TO "postgres";

-- ----------------------------
-- Function structure for dropgeometrytable
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."dropgeometrytable"("schema_name" varchar, "table_name" varchar);
CREATE FUNCTION "public"."dropgeometrytable"("schema_name" varchar, "table_name" varchar)
  RETURNS "pg_catalog"."text" AS $BODY$ SELECT public.DropGeometryTable('',$1,$2) $BODY$
  LANGUAGE sql VOLATILE STRICT
  COST 100;
ALTER FUNCTION "public"."dropgeometrytable"("schema_name" varchar, "table_name" varchar) OWNER TO "postgres";

-- ----------------------------
-- Function structure for dropgeometrytable
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."dropgeometrytable"("catalog_name" varchar, "schema_name" varchar, "table_name" varchar);
CREATE FUNCTION "public"."dropgeometrytable"("catalog_name" varchar, "schema_name" varchar, "table_name" varchar)
  RETURNS "pg_catalog"."text" AS $BODY$
DECLARE
	real_schema name;

BEGIN

	IF ( schema_name = '' ) THEN
		SELECT current_schema() into real_schema;
	ELSE
		real_schema = schema_name;
	END IF;

	-- TODO: Should we warn if table doesn't exist probably instead just saying dropped
	-- Remove table
	EXECUTE 'DROP TABLE IF EXISTS '
		|| quote_ident(real_schema) || '.' ||
		quote_ident(table_name) || ' RESTRICT';

	RETURN
		real_schema || '.' ||
		table_name ||' dropped.';

END;
$BODY$
  LANGUAGE plpgsql VOLATILE STRICT
  COST 100;
ALTER FUNCTION "public"."dropgeometrytable"("catalog_name" varchar, "schema_name" varchar, "table_name" varchar) OWNER TO "postgres";

-- ----------------------------
-- Function structure for dropgeometrytable
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."dropgeometrytable"("table_name" varchar);
CREATE FUNCTION "public"."dropgeometrytable"("table_name" varchar)
  RETURNS "pg_catalog"."text" AS $BODY$ SELECT public.DropGeometryTable('','',$1) $BODY$
  LANGUAGE sql VOLATILE STRICT
  COST 100;
ALTER FUNCTION "public"."dropgeometrytable"("table_name" varchar) OWNER TO "postgres";

-- ----------------------------
-- Function structure for equals
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."equals"("geom1" "public"."geometry", "geom2" "public"."geometry");
CREATE FUNCTION "public"."equals"("geom1" "public"."geometry", "geom2" "public"."geometry")
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'ST_Equals'
  LANGUAGE c IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."equals"("geom1" "public"."geometry", "geom2" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for find_srid
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."find_srid"(varchar, varchar, varchar);
CREATE FUNCTION "public"."find_srid"(varchar, varchar, varchar)
  RETURNS "pg_catalog"."int4" AS $BODY$
DECLARE
	schem varchar =  $1;
	tabl varchar = $2;
	sr int4;
BEGIN
-- if the table contains a . and the schema is empty
-- split the table into a schema and a table
-- otherwise drop through to default behavior
	IF ( schem = '' and strpos(tabl,'.') > 0 ) THEN
	 schem = substr(tabl,1,strpos(tabl,'.')-1);
	 tabl = substr(tabl,length(schem)+2);
	END IF;

	select SRID into sr from public.geometry_columns where (f_table_schema = schem or schem = '') and f_table_name = tabl and f_geometry_column = $3;
	IF NOT FOUND THEN
	   RAISE EXCEPTION 'find_srid() - could not find the corresponding SRID - is the geometry registered in the GEOMETRY_COLUMNS table?  Is there an uppercase/lowercase mismatch?';
	END IF;
	return sr;
END;
$BODY$
  LANGUAGE plpgsql STABLE STRICT
  COST 100;
ALTER FUNCTION "public"."find_srid"(varchar, varchar, varchar) OWNER TO "postgres";

-- ----------------------------
-- Function structure for geog_brin_inclusion_add_value
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geog_brin_inclusion_add_value"(internal, internal, internal, internal);
CREATE FUNCTION "public"."geog_brin_inclusion_add_value"(internal, internal, internal, internal)
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'geog_brin_inclusion_add_value'
  LANGUAGE c VOLATILE
  COST 1;
ALTER FUNCTION "public"."geog_brin_inclusion_add_value"(internal, internal, internal, internal) OWNER TO "postgres";

-- ----------------------------
-- Function structure for geog_brin_inclusion_merge
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geog_brin_inclusion_merge"(internal, internal);
CREATE FUNCTION "public"."geog_brin_inclusion_merge"(internal, internal)
  RETURNS "pg_catalog"."internal" AS '$libdir/postgis-3', 'geog_brin_inclusion_merge'
  LANGUAGE c VOLATILE
  COST 1;
ALTER FUNCTION "public"."geog_brin_inclusion_merge"(internal, internal) OWNER TO "postgres";

-- ----------------------------
-- Function structure for geography
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geography"("public"."geography", int4, bool);
CREATE FUNCTION "public"."geography"("public"."geography", int4, bool)
  RETURNS "public"."geography" AS '$libdir/postgis-3', 'geography_enforce_typmod'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."geography"("public"."geography", int4, bool) OWNER TO "postgres";

-- ----------------------------
-- Function structure for geography
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geography"(bytea);
CREATE FUNCTION "public"."geography"(bytea)
  RETURNS "public"."geography" AS '$libdir/postgis-3', 'geography_from_binary'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."geography"(bytea) OWNER TO "postgres";

-- ----------------------------
-- Function structure for geography
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geography"("public"."geometry");
CREATE FUNCTION "public"."geography"("public"."geometry")
  RETURNS "public"."geography" AS '$libdir/postgis-3', 'geography_from_geometry'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."geography"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for geography_analyze
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geography_analyze"(internal);
CREATE FUNCTION "public"."geography_analyze"(internal)
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'gserialized_analyze_nd'
  LANGUAGE c VOLATILE STRICT
  COST 1;
ALTER FUNCTION "public"."geography_analyze"(internal) OWNER TO "postgres";

-- ----------------------------
-- Function structure for geography_cmp
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geography_cmp"("public"."geography", "public"."geography");
CREATE FUNCTION "public"."geography_cmp"("public"."geography", "public"."geography")
  RETURNS "pg_catalog"."int4" AS '$libdir/postgis-3', 'geography_cmp'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."geography_cmp"("public"."geography", "public"."geography") OWNER TO "postgres";

-- ----------------------------
-- Function structure for geography_distance_knn
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geography_distance_knn"("public"."geography", "public"."geography");
CREATE FUNCTION "public"."geography_distance_knn"("public"."geography", "public"."geography")
  RETURNS "pg_catalog"."float8" AS '$libdir/postgis-3', 'geography_distance_knn'
  LANGUAGE c IMMUTABLE STRICT
  COST 100;
ALTER FUNCTION "public"."geography_distance_knn"("public"."geography", "public"."geography") OWNER TO "postgres";

-- ----------------------------
-- Function structure for geography_eq
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geography_eq"("public"."geography", "public"."geography");
CREATE FUNCTION "public"."geography_eq"("public"."geography", "public"."geography")
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'geography_eq'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."geography_eq"("public"."geography", "public"."geography") OWNER TO "postgres";

-- ----------------------------
-- Function structure for geography_ge
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geography_ge"("public"."geography", "public"."geography");
CREATE FUNCTION "public"."geography_ge"("public"."geography", "public"."geography")
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'geography_ge'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."geography_ge"("public"."geography", "public"."geography") OWNER TO "postgres";

-- ----------------------------
-- Function structure for geography_gist_compress
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geography_gist_compress"(internal);
CREATE FUNCTION "public"."geography_gist_compress"(internal)
  RETURNS "pg_catalog"."internal" AS '$libdir/postgis-3', 'gserialized_gist_compress'
  LANGUAGE c VOLATILE
  COST 1;
ALTER FUNCTION "public"."geography_gist_compress"(internal) OWNER TO "postgres";

-- ----------------------------
-- Function structure for geography_gist_consistent
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geography_gist_consistent"(internal, "public"."geography", int4);
CREATE FUNCTION "public"."geography_gist_consistent"(internal, "public"."geography", int4)
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'gserialized_gist_consistent'
  LANGUAGE c VOLATILE
  COST 1;
ALTER FUNCTION "public"."geography_gist_consistent"(internal, "public"."geography", int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for geography_gist_decompress
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geography_gist_decompress"(internal);
CREATE FUNCTION "public"."geography_gist_decompress"(internal)
  RETURNS "pg_catalog"."internal" AS '$libdir/postgis-3', 'gserialized_gist_decompress'
  LANGUAGE c VOLATILE
  COST 1;
ALTER FUNCTION "public"."geography_gist_decompress"(internal) OWNER TO "postgres";

-- ----------------------------
-- Function structure for geography_gist_distance
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geography_gist_distance"(internal, "public"."geography", int4);
CREATE FUNCTION "public"."geography_gist_distance"(internal, "public"."geography", int4)
  RETURNS "pg_catalog"."float8" AS '$libdir/postgis-3', 'gserialized_gist_geog_distance'
  LANGUAGE c VOLATILE
  COST 1;
ALTER FUNCTION "public"."geography_gist_distance"(internal, "public"."geography", int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for geography_gist_penalty
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geography_gist_penalty"(internal, internal, internal);
CREATE FUNCTION "public"."geography_gist_penalty"(internal, internal, internal)
  RETURNS "pg_catalog"."internal" AS '$libdir/postgis-3', 'gserialized_gist_penalty'
  LANGUAGE c VOLATILE
  COST 1;
ALTER FUNCTION "public"."geography_gist_penalty"(internal, internal, internal) OWNER TO "postgres";

-- ----------------------------
-- Function structure for geography_gist_picksplit
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geography_gist_picksplit"(internal, internal);
CREATE FUNCTION "public"."geography_gist_picksplit"(internal, internal)
  RETURNS "pg_catalog"."internal" AS '$libdir/postgis-3', 'gserialized_gist_picksplit'
  LANGUAGE c VOLATILE
  COST 1;
ALTER FUNCTION "public"."geography_gist_picksplit"(internal, internal) OWNER TO "postgres";

-- ----------------------------
-- Function structure for geography_gist_same
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geography_gist_same"("public"."box2d", "public"."box2d", internal);
CREATE FUNCTION "public"."geography_gist_same"("public"."box2d", "public"."box2d", internal)
  RETURNS "pg_catalog"."internal" AS '$libdir/postgis-3', 'gserialized_gist_same'
  LANGUAGE c VOLATILE
  COST 1;
ALTER FUNCTION "public"."geography_gist_same"("public"."box2d", "public"."box2d", internal) OWNER TO "postgres";

-- ----------------------------
-- Function structure for geography_gist_union
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geography_gist_union"(bytea, internal);
CREATE FUNCTION "public"."geography_gist_union"(bytea, internal)
  RETURNS "pg_catalog"."internal" AS '$libdir/postgis-3', 'gserialized_gist_union'
  LANGUAGE c VOLATILE
  COST 1;
ALTER FUNCTION "public"."geography_gist_union"(bytea, internal) OWNER TO "postgres";

-- ----------------------------
-- Function structure for geography_gt
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geography_gt"("public"."geography", "public"."geography");
CREATE FUNCTION "public"."geography_gt"("public"."geography", "public"."geography")
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'geography_gt'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."geography_gt"("public"."geography", "public"."geography") OWNER TO "postgres";

-- ----------------------------
-- Function structure for geography_in
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geography_in"(cstring, oid, int4);
CREATE FUNCTION "public"."geography_in"(cstring, oid, int4)
  RETURNS "public"."geography" AS '$libdir/postgis-3', 'geography_in'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."geography_in"(cstring, oid, int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for geography_le
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geography_le"("public"."geography", "public"."geography");
CREATE FUNCTION "public"."geography_le"("public"."geography", "public"."geography")
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'geography_le'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."geography_le"("public"."geography", "public"."geography") OWNER TO "postgres";

-- ----------------------------
-- Function structure for geography_lt
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geography_lt"("public"."geography", "public"."geography");
CREATE FUNCTION "public"."geography_lt"("public"."geography", "public"."geography")
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'geography_lt'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."geography_lt"("public"."geography", "public"."geography") OWNER TO "postgres";

-- ----------------------------
-- Function structure for geography_out
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geography_out"("public"."geography");
CREATE FUNCTION "public"."geography_out"("public"."geography")
  RETURNS "pg_catalog"."cstring" AS '$libdir/postgis-3', 'geography_out'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."geography_out"("public"."geography") OWNER TO "postgres";

-- ----------------------------
-- Function structure for geography_overlaps
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geography_overlaps"("public"."geography", "public"."geography");
CREATE FUNCTION "public"."geography_overlaps"("public"."geography", "public"."geography")
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'gserialized_overlaps'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."geography_overlaps"("public"."geography", "public"."geography") OWNER TO "postgres";

-- ----------------------------
-- Function structure for geography_recv
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geography_recv"(internal, oid, int4);
CREATE FUNCTION "public"."geography_recv"(internal, oid, int4)
  RETURNS "public"."geography" AS '$libdir/postgis-3', 'geography_recv'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."geography_recv"(internal, oid, int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for geography_send
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geography_send"("public"."geography");
CREATE FUNCTION "public"."geography_send"("public"."geography")
  RETURNS "pg_catalog"."bytea" AS '$libdir/postgis-3', 'geography_send'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."geography_send"("public"."geography") OWNER TO "postgres";

-- ----------------------------
-- Function structure for geography_spgist_choose_nd
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geography_spgist_choose_nd"(internal, internal);
CREATE FUNCTION "public"."geography_spgist_choose_nd"(internal, internal)
  RETURNS "pg_catalog"."void" AS '$libdir/postgis-3', 'gserialized_spgist_choose_nd'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."geography_spgist_choose_nd"(internal, internal) OWNER TO "postgres";

-- ----------------------------
-- Function structure for geography_spgist_compress_nd
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geography_spgist_compress_nd"(internal);
CREATE FUNCTION "public"."geography_spgist_compress_nd"(internal)
  RETURNS "pg_catalog"."internal" AS '$libdir/postgis-3', 'gserialized_spgist_compress_nd'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."geography_spgist_compress_nd"(internal) OWNER TO "postgres";

-- ----------------------------
-- Function structure for geography_spgist_config_nd
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geography_spgist_config_nd"(internal, internal);
CREATE FUNCTION "public"."geography_spgist_config_nd"(internal, internal)
  RETURNS "pg_catalog"."void" AS '$libdir/postgis-3', 'gserialized_spgist_config_nd'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."geography_spgist_config_nd"(internal, internal) OWNER TO "postgres";

-- ----------------------------
-- Function structure for geography_spgist_inner_consistent_nd
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geography_spgist_inner_consistent_nd"(internal, internal);
CREATE FUNCTION "public"."geography_spgist_inner_consistent_nd"(internal, internal)
  RETURNS "pg_catalog"."void" AS '$libdir/postgis-3', 'gserialized_spgist_inner_consistent_nd'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."geography_spgist_inner_consistent_nd"(internal, internal) OWNER TO "postgres";

-- ----------------------------
-- Function structure for geography_spgist_leaf_consistent_nd
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geography_spgist_leaf_consistent_nd"(internal, internal);
CREATE FUNCTION "public"."geography_spgist_leaf_consistent_nd"(internal, internal)
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'gserialized_spgist_leaf_consistent_nd'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."geography_spgist_leaf_consistent_nd"(internal, internal) OWNER TO "postgres";

-- ----------------------------
-- Function structure for geography_spgist_picksplit_nd
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geography_spgist_picksplit_nd"(internal, internal);
CREATE FUNCTION "public"."geography_spgist_picksplit_nd"(internal, internal)
  RETURNS "pg_catalog"."void" AS '$libdir/postgis-3', 'gserialized_spgist_picksplit_nd'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."geography_spgist_picksplit_nd"(internal, internal) OWNER TO "postgres";

-- ----------------------------
-- Function structure for geography_typmod_in
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geography_typmod_in"(_cstring);
CREATE FUNCTION "public"."geography_typmod_in"(_cstring)
  RETURNS "pg_catalog"."int4" AS '$libdir/postgis-3', 'geography_typmod_in'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."geography_typmod_in"(_cstring) OWNER TO "postgres";

-- ----------------------------
-- Function structure for geography_typmod_out
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geography_typmod_out"(int4);
CREATE FUNCTION "public"."geography_typmod_out"(int4)
  RETURNS "pg_catalog"."cstring" AS '$libdir/postgis-3', 'postgis_typmod_out'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."geography_typmod_out"(int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for geom2d_brin_inclusion_add_value
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geom2d_brin_inclusion_add_value"(internal, internal, internal, internal);
CREATE FUNCTION "public"."geom2d_brin_inclusion_add_value"(internal, internal, internal, internal)
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'geom2d_brin_inclusion_add_value'
  LANGUAGE c VOLATILE
  COST 1;
ALTER FUNCTION "public"."geom2d_brin_inclusion_add_value"(internal, internal, internal, internal) OWNER TO "postgres";

-- ----------------------------
-- Function structure for geom2d_brin_inclusion_merge
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geom2d_brin_inclusion_merge"(internal, internal);
CREATE FUNCTION "public"."geom2d_brin_inclusion_merge"(internal, internal)
  RETURNS "pg_catalog"."internal" AS '$libdir/postgis-3', 'geom2d_brin_inclusion_merge'
  LANGUAGE c VOLATILE
  COST 1;
ALTER FUNCTION "public"."geom2d_brin_inclusion_merge"(internal, internal) OWNER TO "postgres";

-- ----------------------------
-- Function structure for geom3d_brin_inclusion_add_value
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geom3d_brin_inclusion_add_value"(internal, internal, internal, internal);
CREATE FUNCTION "public"."geom3d_brin_inclusion_add_value"(internal, internal, internal, internal)
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'geom3d_brin_inclusion_add_value'
  LANGUAGE c VOLATILE
  COST 1;
ALTER FUNCTION "public"."geom3d_brin_inclusion_add_value"(internal, internal, internal, internal) OWNER TO "postgres";

-- ----------------------------
-- Function structure for geom3d_brin_inclusion_merge
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geom3d_brin_inclusion_merge"(internal, internal);
CREATE FUNCTION "public"."geom3d_brin_inclusion_merge"(internal, internal)
  RETURNS "pg_catalog"."internal" AS '$libdir/postgis-3', 'geom3d_brin_inclusion_merge'
  LANGUAGE c VOLATILE
  COST 1;
ALTER FUNCTION "public"."geom3d_brin_inclusion_merge"(internal, internal) OWNER TO "postgres";

-- ----------------------------
-- Function structure for geom4d_brin_inclusion_add_value
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geom4d_brin_inclusion_add_value"(internal, internal, internal, internal);
CREATE FUNCTION "public"."geom4d_brin_inclusion_add_value"(internal, internal, internal, internal)
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'geom4d_brin_inclusion_add_value'
  LANGUAGE c VOLATILE
  COST 1;
ALTER FUNCTION "public"."geom4d_brin_inclusion_add_value"(internal, internal, internal, internal) OWNER TO "postgres";

-- ----------------------------
-- Function structure for geom4d_brin_inclusion_merge
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geom4d_brin_inclusion_merge"(internal, internal);
CREATE FUNCTION "public"."geom4d_brin_inclusion_merge"(internal, internal)
  RETURNS "pg_catalog"."internal" AS '$libdir/postgis-3', 'geom4d_brin_inclusion_merge'
  LANGUAGE c VOLATILE
  COST 1;
ALTER FUNCTION "public"."geom4d_brin_inclusion_merge"(internal, internal) OWNER TO "postgres";

-- ----------------------------
-- Function structure for geometry
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geometry"(text);
CREATE FUNCTION "public"."geometry"(text)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'parse_WKT_lwgeom'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."geometry"(text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for geometry
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geometry"("public"."box2d");
CREATE FUNCTION "public"."geometry"("public"."box2d")
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'BOX2D_to_LWGEOM'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."geometry"("public"."box2d") OWNER TO "postgres";

-- ----------------------------
-- Function structure for geometry
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geometry"("public"."box3d");
CREATE FUNCTION "public"."geometry"("public"."box3d")
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'BOX3D_to_LWGEOM'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."geometry"("public"."box3d") OWNER TO "postgres";

-- ----------------------------
-- Function structure for geometry
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geometry"(bytea);
CREATE FUNCTION "public"."geometry"(bytea)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'LWGEOM_from_bytea'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."geometry"(bytea) OWNER TO "postgres";

-- ----------------------------
-- Function structure for geometry
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geometry"("public"."geometry", int4, bool);
CREATE FUNCTION "public"."geometry"("public"."geometry", int4, bool)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'geometry_enforce_typmod'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."geometry"("public"."geometry", int4, bool) OWNER TO "postgres";

-- ----------------------------
-- Function structure for geometry
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geometry"(point);
CREATE FUNCTION "public"."geometry"(point)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'point_to_geometry'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."geometry"(point) OWNER TO "postgres";

-- ----------------------------
-- Function structure for geometry
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geometry"(path);
CREATE FUNCTION "public"."geometry"(path)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'path_to_geometry'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."geometry"(path) OWNER TO "postgres";

-- ----------------------------
-- Function structure for geometry
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geometry"(polygon);
CREATE FUNCTION "public"."geometry"(polygon)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'polygon_to_geometry'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."geometry"(polygon) OWNER TO "postgres";

-- ----------------------------
-- Function structure for geometry
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geometry"("public"."geography");
CREATE FUNCTION "public"."geometry"("public"."geography")
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'geometry_from_geography'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."geometry"("public"."geography") OWNER TO "postgres";

-- ----------------------------
-- Function structure for geometry_above
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geometry_above"("geom1" "public"."geometry", "geom2" "public"."geometry");
CREATE FUNCTION "public"."geometry_above"("geom1" "public"."geometry", "geom2" "public"."geometry")
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'gserialized_above_2d'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."geometry_above"("geom1" "public"."geometry", "geom2" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for geometry_analyze
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geometry_analyze"(internal);
CREATE FUNCTION "public"."geometry_analyze"(internal)
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'gserialized_analyze_nd'
  LANGUAGE c VOLATILE STRICT
  COST 1;
ALTER FUNCTION "public"."geometry_analyze"(internal) OWNER TO "postgres";

-- ----------------------------
-- Function structure for geometry_below
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geometry_below"("geom1" "public"."geometry", "geom2" "public"."geometry");
CREATE FUNCTION "public"."geometry_below"("geom1" "public"."geometry", "geom2" "public"."geometry")
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'gserialized_below_2d'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."geometry_below"("geom1" "public"."geometry", "geom2" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for geometry_cmp
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geometry_cmp"("geom1" "public"."geometry", "geom2" "public"."geometry");
CREATE FUNCTION "public"."geometry_cmp"("geom1" "public"."geometry", "geom2" "public"."geometry")
  RETURNS "pg_catalog"."int4" AS '$libdir/postgis-3', 'lwgeom_cmp'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."geometry_cmp"("geom1" "public"."geometry", "geom2" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for geometry_contained_3d
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geometry_contained_3d"("geom1" "public"."geometry", "geom2" "public"."geometry");
CREATE FUNCTION "public"."geometry_contained_3d"("geom1" "public"."geometry", "geom2" "public"."geometry")
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'gserialized_contained_3d'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."geometry_contained_3d"("geom1" "public"."geometry", "geom2" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for geometry_contains
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geometry_contains"("geom1" "public"."geometry", "geom2" "public"."geometry");
CREATE FUNCTION "public"."geometry_contains"("geom1" "public"."geometry", "geom2" "public"."geometry")
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'gserialized_contains_2d'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."geometry_contains"("geom1" "public"."geometry", "geom2" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for geometry_contains_3d
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geometry_contains_3d"("geom1" "public"."geometry", "geom2" "public"."geometry");
CREATE FUNCTION "public"."geometry_contains_3d"("geom1" "public"."geometry", "geom2" "public"."geometry")
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'gserialized_contains_3d'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."geometry_contains_3d"("geom1" "public"."geometry", "geom2" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for geometry_contains_nd
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geometry_contains_nd"("public"."geometry", "public"."geometry");
CREATE FUNCTION "public"."geometry_contains_nd"("public"."geometry", "public"."geometry")
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'gserialized_contains'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."geometry_contains_nd"("public"."geometry", "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for geometry_distance_box
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geometry_distance_box"("geom1" "public"."geometry", "geom2" "public"."geometry");
CREATE FUNCTION "public"."geometry_distance_box"("geom1" "public"."geometry", "geom2" "public"."geometry")
  RETURNS "pg_catalog"."float8" AS '$libdir/postgis-3', 'gserialized_distance_box_2d'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."geometry_distance_box"("geom1" "public"."geometry", "geom2" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for geometry_distance_centroid
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geometry_distance_centroid"("geom1" "public"."geometry", "geom2" "public"."geometry");
CREATE FUNCTION "public"."geometry_distance_centroid"("geom1" "public"."geometry", "geom2" "public"."geometry")
  RETURNS "pg_catalog"."float8" AS '$libdir/postgis-3', 'ST_Distance'
  LANGUAGE c IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."geometry_distance_centroid"("geom1" "public"."geometry", "geom2" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for geometry_distance_centroid_nd
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geometry_distance_centroid_nd"("public"."geometry", "public"."geometry");
CREATE FUNCTION "public"."geometry_distance_centroid_nd"("public"."geometry", "public"."geometry")
  RETURNS "pg_catalog"."float8" AS '$libdir/postgis-3', 'gserialized_distance_nd'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."geometry_distance_centroid_nd"("public"."geometry", "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for geometry_distance_cpa
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geometry_distance_cpa"("public"."geometry", "public"."geometry");
CREATE FUNCTION "public"."geometry_distance_cpa"("public"."geometry", "public"."geometry")
  RETURNS "pg_catalog"."float8" AS '$libdir/postgis-3', 'ST_DistanceCPA'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."geometry_distance_cpa"("public"."geometry", "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for geometry_eq
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geometry_eq"("geom1" "public"."geometry", "geom2" "public"."geometry");
CREATE FUNCTION "public"."geometry_eq"("geom1" "public"."geometry", "geom2" "public"."geometry")
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'lwgeom_eq'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."geometry_eq"("geom1" "public"."geometry", "geom2" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for geometry_ge
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geometry_ge"("geom1" "public"."geometry", "geom2" "public"."geometry");
CREATE FUNCTION "public"."geometry_ge"("geom1" "public"."geometry", "geom2" "public"."geometry")
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'lwgeom_ge'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."geometry_ge"("geom1" "public"."geometry", "geom2" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for geometry_gist_compress_2d
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geometry_gist_compress_2d"(internal);
CREATE FUNCTION "public"."geometry_gist_compress_2d"(internal)
  RETURNS "pg_catalog"."internal" AS '$libdir/postgis-3', 'gserialized_gist_compress_2d'
  LANGUAGE c VOLATILE
  COST 1;
ALTER FUNCTION "public"."geometry_gist_compress_2d"(internal) OWNER TO "postgres";

-- ----------------------------
-- Function structure for geometry_gist_compress_nd
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geometry_gist_compress_nd"(internal);
CREATE FUNCTION "public"."geometry_gist_compress_nd"(internal)
  RETURNS "pg_catalog"."internal" AS '$libdir/postgis-3', 'gserialized_gist_compress'
  LANGUAGE c VOLATILE
  COST 1;
ALTER FUNCTION "public"."geometry_gist_compress_nd"(internal) OWNER TO "postgres";

-- ----------------------------
-- Function structure for geometry_gist_consistent_2d
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geometry_gist_consistent_2d"(internal, "public"."geometry", int4);
CREATE FUNCTION "public"."geometry_gist_consistent_2d"(internal, "public"."geometry", int4)
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'gserialized_gist_consistent_2d'
  LANGUAGE c VOLATILE
  COST 1;
ALTER FUNCTION "public"."geometry_gist_consistent_2d"(internal, "public"."geometry", int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for geometry_gist_consistent_nd
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geometry_gist_consistent_nd"(internal, "public"."geometry", int4);
CREATE FUNCTION "public"."geometry_gist_consistent_nd"(internal, "public"."geometry", int4)
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'gserialized_gist_consistent'
  LANGUAGE c VOLATILE
  COST 1;
ALTER FUNCTION "public"."geometry_gist_consistent_nd"(internal, "public"."geometry", int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for geometry_gist_decompress_2d
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geometry_gist_decompress_2d"(internal);
CREATE FUNCTION "public"."geometry_gist_decompress_2d"(internal)
  RETURNS "pg_catalog"."internal" AS '$libdir/postgis-3', 'gserialized_gist_decompress_2d'
  LANGUAGE c VOLATILE
  COST 1;
ALTER FUNCTION "public"."geometry_gist_decompress_2d"(internal) OWNER TO "postgres";

-- ----------------------------
-- Function structure for geometry_gist_decompress_nd
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geometry_gist_decompress_nd"(internal);
CREATE FUNCTION "public"."geometry_gist_decompress_nd"(internal)
  RETURNS "pg_catalog"."internal" AS '$libdir/postgis-3', 'gserialized_gist_decompress'
  LANGUAGE c VOLATILE
  COST 1;
ALTER FUNCTION "public"."geometry_gist_decompress_nd"(internal) OWNER TO "postgres";

-- ----------------------------
-- Function structure for geometry_gist_distance_2d
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geometry_gist_distance_2d"(internal, "public"."geometry", int4);
CREATE FUNCTION "public"."geometry_gist_distance_2d"(internal, "public"."geometry", int4)
  RETURNS "pg_catalog"."float8" AS '$libdir/postgis-3', 'gserialized_gist_distance_2d'
  LANGUAGE c VOLATILE
  COST 1;
ALTER FUNCTION "public"."geometry_gist_distance_2d"(internal, "public"."geometry", int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for geometry_gist_distance_nd
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geometry_gist_distance_nd"(internal, "public"."geometry", int4);
CREATE FUNCTION "public"."geometry_gist_distance_nd"(internal, "public"."geometry", int4)
  RETURNS "pg_catalog"."float8" AS '$libdir/postgis-3', 'gserialized_gist_distance'
  LANGUAGE c VOLATILE
  COST 1;
ALTER FUNCTION "public"."geometry_gist_distance_nd"(internal, "public"."geometry", int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for geometry_gist_penalty_2d
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geometry_gist_penalty_2d"(internal, internal, internal);
CREATE FUNCTION "public"."geometry_gist_penalty_2d"(internal, internal, internal)
  RETURNS "pg_catalog"."internal" AS '$libdir/postgis-3', 'gserialized_gist_penalty_2d'
  LANGUAGE c VOLATILE
  COST 1;
ALTER FUNCTION "public"."geometry_gist_penalty_2d"(internal, internal, internal) OWNER TO "postgres";

-- ----------------------------
-- Function structure for geometry_gist_penalty_nd
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geometry_gist_penalty_nd"(internal, internal, internal);
CREATE FUNCTION "public"."geometry_gist_penalty_nd"(internal, internal, internal)
  RETURNS "pg_catalog"."internal" AS '$libdir/postgis-3', 'gserialized_gist_penalty'
  LANGUAGE c VOLATILE
  COST 1;
ALTER FUNCTION "public"."geometry_gist_penalty_nd"(internal, internal, internal) OWNER TO "postgres";

-- ----------------------------
-- Function structure for geometry_gist_picksplit_2d
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geometry_gist_picksplit_2d"(internal, internal);
CREATE FUNCTION "public"."geometry_gist_picksplit_2d"(internal, internal)
  RETURNS "pg_catalog"."internal" AS '$libdir/postgis-3', 'gserialized_gist_picksplit_2d'
  LANGUAGE c VOLATILE
  COST 1;
ALTER FUNCTION "public"."geometry_gist_picksplit_2d"(internal, internal) OWNER TO "postgres";

-- ----------------------------
-- Function structure for geometry_gist_picksplit_nd
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geometry_gist_picksplit_nd"(internal, internal);
CREATE FUNCTION "public"."geometry_gist_picksplit_nd"(internal, internal)
  RETURNS "pg_catalog"."internal" AS '$libdir/postgis-3', 'gserialized_gist_picksplit'
  LANGUAGE c VOLATILE
  COST 1;
ALTER FUNCTION "public"."geometry_gist_picksplit_nd"(internal, internal) OWNER TO "postgres";

-- ----------------------------
-- Function structure for geometry_gist_same_2d
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geometry_gist_same_2d"("geom1" "public"."geometry", "geom2" "public"."geometry", internal);
CREATE FUNCTION "public"."geometry_gist_same_2d"("geom1" "public"."geometry", "geom2" "public"."geometry", internal)
  RETURNS "pg_catalog"."internal" AS '$libdir/postgis-3', 'gserialized_gist_same_2d'
  LANGUAGE c VOLATILE
  COST 1;
ALTER FUNCTION "public"."geometry_gist_same_2d"("geom1" "public"."geometry", "geom2" "public"."geometry", internal) OWNER TO "postgres";

-- ----------------------------
-- Function structure for geometry_gist_same_nd
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geometry_gist_same_nd"("public"."geometry", "public"."geometry", internal);
CREATE FUNCTION "public"."geometry_gist_same_nd"("public"."geometry", "public"."geometry", internal)
  RETURNS "pg_catalog"."internal" AS '$libdir/postgis-3', 'gserialized_gist_same'
  LANGUAGE c VOLATILE
  COST 1;
ALTER FUNCTION "public"."geometry_gist_same_nd"("public"."geometry", "public"."geometry", internal) OWNER TO "postgres";

-- ----------------------------
-- Function structure for geometry_gist_sortsupport_2d
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geometry_gist_sortsupport_2d"(internal);
CREATE FUNCTION "public"."geometry_gist_sortsupport_2d"(internal)
  RETURNS "pg_catalog"."void" AS '$libdir/postgis-3', 'gserialized_gist_sortsupport_2d'
  LANGUAGE c VOLATILE STRICT
  COST 1;
ALTER FUNCTION "public"."geometry_gist_sortsupport_2d"(internal) OWNER TO "postgres";

-- ----------------------------
-- Function structure for geometry_gist_union_2d
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geometry_gist_union_2d"(bytea, internal);
CREATE FUNCTION "public"."geometry_gist_union_2d"(bytea, internal)
  RETURNS "pg_catalog"."internal" AS '$libdir/postgis-3', 'gserialized_gist_union_2d'
  LANGUAGE c VOLATILE
  COST 1;
ALTER FUNCTION "public"."geometry_gist_union_2d"(bytea, internal) OWNER TO "postgres";

-- ----------------------------
-- Function structure for geometry_gist_union_nd
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geometry_gist_union_nd"(bytea, internal);
CREATE FUNCTION "public"."geometry_gist_union_nd"(bytea, internal)
  RETURNS "pg_catalog"."internal" AS '$libdir/postgis-3', 'gserialized_gist_union'
  LANGUAGE c VOLATILE
  COST 1;
ALTER FUNCTION "public"."geometry_gist_union_nd"(bytea, internal) OWNER TO "postgres";

-- ----------------------------
-- Function structure for geometry_gt
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geometry_gt"("geom1" "public"."geometry", "geom2" "public"."geometry");
CREATE FUNCTION "public"."geometry_gt"("geom1" "public"."geometry", "geom2" "public"."geometry")
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'lwgeom_gt'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."geometry_gt"("geom1" "public"."geometry", "geom2" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for geometry_hash
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geometry_hash"("public"."geometry");
CREATE FUNCTION "public"."geometry_hash"("public"."geometry")
  RETURNS "pg_catalog"."int4" AS '$libdir/postgis-3', 'lwgeom_hash'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."geometry_hash"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for geometry_in
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geometry_in"(cstring);
CREATE FUNCTION "public"."geometry_in"(cstring)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'LWGEOM_in'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."geometry_in"(cstring) OWNER TO "postgres";

-- ----------------------------
-- Function structure for geometry_le
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geometry_le"("geom1" "public"."geometry", "geom2" "public"."geometry");
CREATE FUNCTION "public"."geometry_le"("geom1" "public"."geometry", "geom2" "public"."geometry")
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'lwgeom_le'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."geometry_le"("geom1" "public"."geometry", "geom2" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for geometry_left
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geometry_left"("geom1" "public"."geometry", "geom2" "public"."geometry");
CREATE FUNCTION "public"."geometry_left"("geom1" "public"."geometry", "geom2" "public"."geometry")
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'gserialized_left_2d'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."geometry_left"("geom1" "public"."geometry", "geom2" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for geometry_lt
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geometry_lt"("geom1" "public"."geometry", "geom2" "public"."geometry");
CREATE FUNCTION "public"."geometry_lt"("geom1" "public"."geometry", "geom2" "public"."geometry")
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'lwgeom_lt'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."geometry_lt"("geom1" "public"."geometry", "geom2" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for geometry_neq
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geometry_neq"("geom1" "public"."geometry", "geom2" "public"."geometry");
CREATE FUNCTION "public"."geometry_neq"("geom1" "public"."geometry", "geom2" "public"."geometry")
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'lwgeom_neq'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."geometry_neq"("geom1" "public"."geometry", "geom2" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for geometry_out
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geometry_out"("public"."geometry");
CREATE FUNCTION "public"."geometry_out"("public"."geometry")
  RETURNS "pg_catalog"."cstring" AS '$libdir/postgis-3', 'LWGEOM_out'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."geometry_out"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for geometry_overabove
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geometry_overabove"("geom1" "public"."geometry", "geom2" "public"."geometry");
CREATE FUNCTION "public"."geometry_overabove"("geom1" "public"."geometry", "geom2" "public"."geometry")
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'gserialized_overabove_2d'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."geometry_overabove"("geom1" "public"."geometry", "geom2" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for geometry_overbelow
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geometry_overbelow"("geom1" "public"."geometry", "geom2" "public"."geometry");
CREATE FUNCTION "public"."geometry_overbelow"("geom1" "public"."geometry", "geom2" "public"."geometry")
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'gserialized_overbelow_2d'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."geometry_overbelow"("geom1" "public"."geometry", "geom2" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for geometry_overlaps
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geometry_overlaps"("geom1" "public"."geometry", "geom2" "public"."geometry");
CREATE FUNCTION "public"."geometry_overlaps"("geom1" "public"."geometry", "geom2" "public"."geometry")
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'gserialized_overlaps_2d'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."geometry_overlaps"("geom1" "public"."geometry", "geom2" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for geometry_overlaps_3d
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geometry_overlaps_3d"("geom1" "public"."geometry", "geom2" "public"."geometry");
CREATE FUNCTION "public"."geometry_overlaps_3d"("geom1" "public"."geometry", "geom2" "public"."geometry")
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'gserialized_overlaps_3d'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."geometry_overlaps_3d"("geom1" "public"."geometry", "geom2" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for geometry_overlaps_nd
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geometry_overlaps_nd"("public"."geometry", "public"."geometry");
CREATE FUNCTION "public"."geometry_overlaps_nd"("public"."geometry", "public"."geometry")
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'gserialized_overlaps'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."geometry_overlaps_nd"("public"."geometry", "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for geometry_overleft
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geometry_overleft"("geom1" "public"."geometry", "geom2" "public"."geometry");
CREATE FUNCTION "public"."geometry_overleft"("geom1" "public"."geometry", "geom2" "public"."geometry")
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'gserialized_overleft_2d'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."geometry_overleft"("geom1" "public"."geometry", "geom2" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for geometry_overright
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geometry_overright"("geom1" "public"."geometry", "geom2" "public"."geometry");
CREATE FUNCTION "public"."geometry_overright"("geom1" "public"."geometry", "geom2" "public"."geometry")
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'gserialized_overright_2d'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."geometry_overright"("geom1" "public"."geometry", "geom2" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for geometry_recv
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geometry_recv"(internal);
CREATE FUNCTION "public"."geometry_recv"(internal)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'LWGEOM_recv'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."geometry_recv"(internal) OWNER TO "postgres";

-- ----------------------------
-- Function structure for geometry_right
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geometry_right"("geom1" "public"."geometry", "geom2" "public"."geometry");
CREATE FUNCTION "public"."geometry_right"("geom1" "public"."geometry", "geom2" "public"."geometry")
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'gserialized_right_2d'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."geometry_right"("geom1" "public"."geometry", "geom2" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for geometry_same
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geometry_same"("geom1" "public"."geometry", "geom2" "public"."geometry");
CREATE FUNCTION "public"."geometry_same"("geom1" "public"."geometry", "geom2" "public"."geometry")
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'gserialized_same_2d'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."geometry_same"("geom1" "public"."geometry", "geom2" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for geometry_same_3d
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geometry_same_3d"("geom1" "public"."geometry", "geom2" "public"."geometry");
CREATE FUNCTION "public"."geometry_same_3d"("geom1" "public"."geometry", "geom2" "public"."geometry")
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'gserialized_same_3d'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."geometry_same_3d"("geom1" "public"."geometry", "geom2" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for geometry_same_nd
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geometry_same_nd"("public"."geometry", "public"."geometry");
CREATE FUNCTION "public"."geometry_same_nd"("public"."geometry", "public"."geometry")
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'gserialized_same'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."geometry_same_nd"("public"."geometry", "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for geometry_send
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geometry_send"("public"."geometry");
CREATE FUNCTION "public"."geometry_send"("public"."geometry")
  RETURNS "pg_catalog"."bytea" AS '$libdir/postgis-3', 'LWGEOM_send'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."geometry_send"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for geometry_sortsupport
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geometry_sortsupport"(internal);
CREATE FUNCTION "public"."geometry_sortsupport"(internal)
  RETURNS "pg_catalog"."void" AS '$libdir/postgis-3', 'lwgeom_sortsupport'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."geometry_sortsupport"(internal) OWNER TO "postgres";

-- ----------------------------
-- Function structure for geometry_spgist_choose_2d
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geometry_spgist_choose_2d"(internal, internal);
CREATE FUNCTION "public"."geometry_spgist_choose_2d"(internal, internal)
  RETURNS "pg_catalog"."void" AS '$libdir/postgis-3', 'gserialized_spgist_choose_2d'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."geometry_spgist_choose_2d"(internal, internal) OWNER TO "postgres";

-- ----------------------------
-- Function structure for geometry_spgist_choose_3d
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geometry_spgist_choose_3d"(internal, internal);
CREATE FUNCTION "public"."geometry_spgist_choose_3d"(internal, internal)
  RETURNS "pg_catalog"."void" AS '$libdir/postgis-3', 'gserialized_spgist_choose_3d'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."geometry_spgist_choose_3d"(internal, internal) OWNER TO "postgres";

-- ----------------------------
-- Function structure for geometry_spgist_choose_nd
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geometry_spgist_choose_nd"(internal, internal);
CREATE FUNCTION "public"."geometry_spgist_choose_nd"(internal, internal)
  RETURNS "pg_catalog"."void" AS '$libdir/postgis-3', 'gserialized_spgist_choose_nd'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."geometry_spgist_choose_nd"(internal, internal) OWNER TO "postgres";

-- ----------------------------
-- Function structure for geometry_spgist_compress_2d
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geometry_spgist_compress_2d"(internal);
CREATE FUNCTION "public"."geometry_spgist_compress_2d"(internal)
  RETURNS "pg_catalog"."internal" AS '$libdir/postgis-3', 'gserialized_spgist_compress_2d'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."geometry_spgist_compress_2d"(internal) OWNER TO "postgres";

-- ----------------------------
-- Function structure for geometry_spgist_compress_3d
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geometry_spgist_compress_3d"(internal);
CREATE FUNCTION "public"."geometry_spgist_compress_3d"(internal)
  RETURNS "pg_catalog"."internal" AS '$libdir/postgis-3', 'gserialized_spgist_compress_3d'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."geometry_spgist_compress_3d"(internal) OWNER TO "postgres";

-- ----------------------------
-- Function structure for geometry_spgist_compress_nd
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geometry_spgist_compress_nd"(internal);
CREATE FUNCTION "public"."geometry_spgist_compress_nd"(internal)
  RETURNS "pg_catalog"."internal" AS '$libdir/postgis-3', 'gserialized_spgist_compress_nd'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."geometry_spgist_compress_nd"(internal) OWNER TO "postgres";

-- ----------------------------
-- Function structure for geometry_spgist_config_2d
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geometry_spgist_config_2d"(internal, internal);
CREATE FUNCTION "public"."geometry_spgist_config_2d"(internal, internal)
  RETURNS "pg_catalog"."void" AS '$libdir/postgis-3', 'gserialized_spgist_config_2d'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."geometry_spgist_config_2d"(internal, internal) OWNER TO "postgres";

-- ----------------------------
-- Function structure for geometry_spgist_config_3d
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geometry_spgist_config_3d"(internal, internal);
CREATE FUNCTION "public"."geometry_spgist_config_3d"(internal, internal)
  RETURNS "pg_catalog"."void" AS '$libdir/postgis-3', 'gserialized_spgist_config_3d'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."geometry_spgist_config_3d"(internal, internal) OWNER TO "postgres";

-- ----------------------------
-- Function structure for geometry_spgist_config_nd
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geometry_spgist_config_nd"(internal, internal);
CREATE FUNCTION "public"."geometry_spgist_config_nd"(internal, internal)
  RETURNS "pg_catalog"."void" AS '$libdir/postgis-3', 'gserialized_spgist_config_nd'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."geometry_spgist_config_nd"(internal, internal) OWNER TO "postgres";

-- ----------------------------
-- Function structure for geometry_spgist_inner_consistent_2d
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geometry_spgist_inner_consistent_2d"(internal, internal);
CREATE FUNCTION "public"."geometry_spgist_inner_consistent_2d"(internal, internal)
  RETURNS "pg_catalog"."void" AS '$libdir/postgis-3', 'gserialized_spgist_inner_consistent_2d'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."geometry_spgist_inner_consistent_2d"(internal, internal) OWNER TO "postgres";

-- ----------------------------
-- Function structure for geometry_spgist_inner_consistent_3d
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geometry_spgist_inner_consistent_3d"(internal, internal);
CREATE FUNCTION "public"."geometry_spgist_inner_consistent_3d"(internal, internal)
  RETURNS "pg_catalog"."void" AS '$libdir/postgis-3', 'gserialized_spgist_inner_consistent_3d'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."geometry_spgist_inner_consistent_3d"(internal, internal) OWNER TO "postgres";

-- ----------------------------
-- Function structure for geometry_spgist_inner_consistent_nd
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geometry_spgist_inner_consistent_nd"(internal, internal);
CREATE FUNCTION "public"."geometry_spgist_inner_consistent_nd"(internal, internal)
  RETURNS "pg_catalog"."void" AS '$libdir/postgis-3', 'gserialized_spgist_inner_consistent_nd'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."geometry_spgist_inner_consistent_nd"(internal, internal) OWNER TO "postgres";

-- ----------------------------
-- Function structure for geometry_spgist_leaf_consistent_2d
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geometry_spgist_leaf_consistent_2d"(internal, internal);
CREATE FUNCTION "public"."geometry_spgist_leaf_consistent_2d"(internal, internal)
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'gserialized_spgist_leaf_consistent_2d'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."geometry_spgist_leaf_consistent_2d"(internal, internal) OWNER TO "postgres";

-- ----------------------------
-- Function structure for geometry_spgist_leaf_consistent_3d
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geometry_spgist_leaf_consistent_3d"(internal, internal);
CREATE FUNCTION "public"."geometry_spgist_leaf_consistent_3d"(internal, internal)
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'gserialized_spgist_leaf_consistent_3d'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."geometry_spgist_leaf_consistent_3d"(internal, internal) OWNER TO "postgres";

-- ----------------------------
-- Function structure for geometry_spgist_leaf_consistent_nd
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geometry_spgist_leaf_consistent_nd"(internal, internal);
CREATE FUNCTION "public"."geometry_spgist_leaf_consistent_nd"(internal, internal)
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'gserialized_spgist_leaf_consistent_nd'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."geometry_spgist_leaf_consistent_nd"(internal, internal) OWNER TO "postgres";

-- ----------------------------
-- Function structure for geometry_spgist_picksplit_2d
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geometry_spgist_picksplit_2d"(internal, internal);
CREATE FUNCTION "public"."geometry_spgist_picksplit_2d"(internal, internal)
  RETURNS "pg_catalog"."void" AS '$libdir/postgis-3', 'gserialized_spgist_picksplit_2d'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."geometry_spgist_picksplit_2d"(internal, internal) OWNER TO "postgres";

-- ----------------------------
-- Function structure for geometry_spgist_picksplit_3d
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geometry_spgist_picksplit_3d"(internal, internal);
CREATE FUNCTION "public"."geometry_spgist_picksplit_3d"(internal, internal)
  RETURNS "pg_catalog"."void" AS '$libdir/postgis-3', 'gserialized_spgist_picksplit_3d'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."geometry_spgist_picksplit_3d"(internal, internal) OWNER TO "postgres";

-- ----------------------------
-- Function structure for geometry_spgist_picksplit_nd
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geometry_spgist_picksplit_nd"(internal, internal);
CREATE FUNCTION "public"."geometry_spgist_picksplit_nd"(internal, internal)
  RETURNS "pg_catalog"."void" AS '$libdir/postgis-3', 'gserialized_spgist_picksplit_nd'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."geometry_spgist_picksplit_nd"(internal, internal) OWNER TO "postgres";

-- ----------------------------
-- Function structure for geometry_typmod_in
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geometry_typmod_in"(_cstring);
CREATE FUNCTION "public"."geometry_typmod_in"(_cstring)
  RETURNS "pg_catalog"."int4" AS '$libdir/postgis-3', 'geometry_typmod_in'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."geometry_typmod_in"(_cstring) OWNER TO "postgres";

-- ----------------------------
-- Function structure for geometry_typmod_out
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geometry_typmod_out"(int4);
CREATE FUNCTION "public"."geometry_typmod_out"(int4)
  RETURNS "pg_catalog"."cstring" AS '$libdir/postgis-3', 'postgis_typmod_out'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."geometry_typmod_out"(int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for geometry_within
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geometry_within"("geom1" "public"."geometry", "geom2" "public"."geometry");
CREATE FUNCTION "public"."geometry_within"("geom1" "public"."geometry", "geom2" "public"."geometry")
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'gserialized_within_2d'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."geometry_within"("geom1" "public"."geometry", "geom2" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for geometry_within_nd
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geometry_within_nd"("public"."geometry", "public"."geometry");
CREATE FUNCTION "public"."geometry_within_nd"("public"."geometry", "public"."geometry")
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'gserialized_within'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."geometry_within_nd"("public"."geometry", "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for geometrytype
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geometrytype"("public"."geometry");
CREATE FUNCTION "public"."geometrytype"("public"."geometry")
  RETURNS "pg_catalog"."text" AS '$libdir/postgis-3', 'LWGEOM_getTYPE'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."geometrytype"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for geometrytype
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geometrytype"("public"."geography");
CREATE FUNCTION "public"."geometrytype"("public"."geography")
  RETURNS "pg_catalog"."text" AS '$libdir/postgis-3', 'LWGEOM_getTYPE'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."geometrytype"("public"."geography") OWNER TO "postgres";

-- ----------------------------
-- Function structure for geomfromewkb
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geomfromewkb"(bytea);
CREATE FUNCTION "public"."geomfromewkb"(bytea)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'LWGEOMFromEWKB'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."geomfromewkb"(bytea) OWNER TO "postgres";

-- ----------------------------
-- Function structure for geomfromewkt
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."geomfromewkt"(text);
CREATE FUNCTION "public"."geomfromewkt"(text)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'parse_WKT_lwgeom'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."geomfromewkt"(text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for get_proj4_from_srid
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."get_proj4_from_srid"(int4);
CREATE FUNCTION "public"."get_proj4_from_srid"(int4)
  RETURNS "pg_catalog"."text" AS $BODY$
	BEGIN
	RETURN proj4text::text FROM public.spatial_ref_sys WHERE srid= $1;
	END;
	$BODY$
  LANGUAGE plpgsql IMMUTABLE STRICT
  COST 100;
ALTER FUNCTION "public"."get_proj4_from_srid"(int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for gidx_in
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."gidx_in"(cstring);
CREATE FUNCTION "public"."gidx_in"(cstring)
  RETURNS "public"."gidx" AS '$libdir/postgis-3', 'gidx_in'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."gidx_in"(cstring) OWNER TO "postgres";

-- ----------------------------
-- Function structure for gidx_out
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."gidx_out"("public"."gidx");
CREATE FUNCTION "public"."gidx_out"("public"."gidx")
  RETURNS "pg_catalog"."cstring" AS '$libdir/postgis-3', 'gidx_out'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."gidx_out"("public"."gidx") OWNER TO "postgres";

-- ----------------------------
-- Function structure for gin_extract_query_trgm
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."gin_extract_query_trgm"(text, internal, int2, internal, internal, internal, internal);
CREATE FUNCTION "public"."gin_extract_query_trgm"(text, internal, int2, internal, internal, internal, internal)
  RETURNS "pg_catalog"."internal" AS '$libdir/pg_trgm', 'gin_extract_query_trgm'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."gin_extract_query_trgm"(text, internal, int2, internal, internal, internal, internal) OWNER TO "postgres";

-- ----------------------------
-- Function structure for gin_extract_value_trgm
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."gin_extract_value_trgm"(text, internal);
CREATE FUNCTION "public"."gin_extract_value_trgm"(text, internal)
  RETURNS "pg_catalog"."internal" AS '$libdir/pg_trgm', 'gin_extract_value_trgm'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."gin_extract_value_trgm"(text, internal) OWNER TO "postgres";

-- ----------------------------
-- Function structure for gin_trgm_consistent
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."gin_trgm_consistent"(internal, int2, text, int4, internal, internal, internal, internal);
CREATE FUNCTION "public"."gin_trgm_consistent"(internal, int2, text, int4, internal, internal, internal, internal)
  RETURNS "pg_catalog"."bool" AS '$libdir/pg_trgm', 'gin_trgm_consistent'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."gin_trgm_consistent"(internal, int2, text, int4, internal, internal, internal, internal) OWNER TO "postgres";

-- ----------------------------
-- Function structure for gin_trgm_triconsistent
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."gin_trgm_triconsistent"(internal, int2, text, int4, internal, internal, internal);
CREATE FUNCTION "public"."gin_trgm_triconsistent"(internal, int2, text, int4, internal, internal, internal)
  RETURNS "pg_catalog"."char" AS '$libdir/pg_trgm', 'gin_trgm_triconsistent'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."gin_trgm_triconsistent"(internal, int2, text, int4, internal, internal, internal) OWNER TO "postgres";

-- ----------------------------
-- Function structure for gserialized_gist_joinsel_2d
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."gserialized_gist_joinsel_2d"(internal, oid, internal, int2);
CREATE FUNCTION "public"."gserialized_gist_joinsel_2d"(internal, oid, internal, int2)
  RETURNS "pg_catalog"."float8" AS '$libdir/postgis-3', 'gserialized_gist_joinsel_2d'
  LANGUAGE c VOLATILE
  COST 1;
ALTER FUNCTION "public"."gserialized_gist_joinsel_2d"(internal, oid, internal, int2) OWNER TO "postgres";

-- ----------------------------
-- Function structure for gserialized_gist_joinsel_nd
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."gserialized_gist_joinsel_nd"(internal, oid, internal, int2);
CREATE FUNCTION "public"."gserialized_gist_joinsel_nd"(internal, oid, internal, int2)
  RETURNS "pg_catalog"."float8" AS '$libdir/postgis-3', 'gserialized_gist_joinsel_nd'
  LANGUAGE c VOLATILE
  COST 1;
ALTER FUNCTION "public"."gserialized_gist_joinsel_nd"(internal, oid, internal, int2) OWNER TO "postgres";

-- ----------------------------
-- Function structure for gserialized_gist_sel_2d
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."gserialized_gist_sel_2d"(internal, oid, internal, int4);
CREATE FUNCTION "public"."gserialized_gist_sel_2d"(internal, oid, internal, int4)
  RETURNS "pg_catalog"."float8" AS '$libdir/postgis-3', 'gserialized_gist_sel_2d'
  LANGUAGE c VOLATILE
  COST 1;
ALTER FUNCTION "public"."gserialized_gist_sel_2d"(internal, oid, internal, int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for gserialized_gist_sel_nd
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."gserialized_gist_sel_nd"(internal, oid, internal, int4);
CREATE FUNCTION "public"."gserialized_gist_sel_nd"(internal, oid, internal, int4)
  RETURNS "pg_catalog"."float8" AS '$libdir/postgis-3', 'gserialized_gist_sel_nd'
  LANGUAGE c VOLATILE
  COST 1;
ALTER FUNCTION "public"."gserialized_gist_sel_nd"(internal, oid, internal, int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for gtrgm_compress
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."gtrgm_compress"(internal);
CREATE FUNCTION "public"."gtrgm_compress"(internal)
  RETURNS "pg_catalog"."internal" AS '$libdir/pg_trgm', 'gtrgm_compress'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."gtrgm_compress"(internal) OWNER TO "postgres";

-- ----------------------------
-- Function structure for gtrgm_consistent
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."gtrgm_consistent"(internal, text, int2, oid, internal);
CREATE FUNCTION "public"."gtrgm_consistent"(internal, text, int2, oid, internal)
  RETURNS "pg_catalog"."bool" AS '$libdir/pg_trgm', 'gtrgm_consistent'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."gtrgm_consistent"(internal, text, int2, oid, internal) OWNER TO "postgres";

-- ----------------------------
-- Function structure for gtrgm_decompress
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."gtrgm_decompress"(internal);
CREATE FUNCTION "public"."gtrgm_decompress"(internal)
  RETURNS "pg_catalog"."internal" AS '$libdir/pg_trgm', 'gtrgm_decompress'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."gtrgm_decompress"(internal) OWNER TO "postgres";

-- ----------------------------
-- Function structure for gtrgm_distance
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."gtrgm_distance"(internal, text, int2, oid, internal);
CREATE FUNCTION "public"."gtrgm_distance"(internal, text, int2, oid, internal)
  RETURNS "pg_catalog"."float8" AS '$libdir/pg_trgm', 'gtrgm_distance'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."gtrgm_distance"(internal, text, int2, oid, internal) OWNER TO "postgres";

-- ----------------------------
-- Function structure for gtrgm_in
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."gtrgm_in"(cstring);
CREATE FUNCTION "public"."gtrgm_in"(cstring)
  RETURNS "public"."gtrgm" AS '$libdir/pg_trgm', 'gtrgm_in'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."gtrgm_in"(cstring) OWNER TO "postgres";

-- ----------------------------
-- Function structure for gtrgm_options
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."gtrgm_options"(internal);
CREATE FUNCTION "public"."gtrgm_options"(internal)
  RETURNS "pg_catalog"."void" AS '$libdir/pg_trgm', 'gtrgm_options'
  LANGUAGE c IMMUTABLE
  COST 1;
ALTER FUNCTION "public"."gtrgm_options"(internal) OWNER TO "postgres";

-- ----------------------------
-- Function structure for gtrgm_out
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."gtrgm_out"("public"."gtrgm");
CREATE FUNCTION "public"."gtrgm_out"("public"."gtrgm")
  RETURNS "pg_catalog"."cstring" AS '$libdir/pg_trgm', 'gtrgm_out'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."gtrgm_out"("public"."gtrgm") OWNER TO "postgres";

-- ----------------------------
-- Function structure for gtrgm_penalty
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."gtrgm_penalty"(internal, internal, internal);
CREATE FUNCTION "public"."gtrgm_penalty"(internal, internal, internal)
  RETURNS "pg_catalog"."internal" AS '$libdir/pg_trgm', 'gtrgm_penalty'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."gtrgm_penalty"(internal, internal, internal) OWNER TO "postgres";

-- ----------------------------
-- Function structure for gtrgm_picksplit
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."gtrgm_picksplit"(internal, internal);
CREATE FUNCTION "public"."gtrgm_picksplit"(internal, internal)
  RETURNS "pg_catalog"."internal" AS '$libdir/pg_trgm', 'gtrgm_picksplit'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."gtrgm_picksplit"(internal, internal) OWNER TO "postgres";

-- ----------------------------
-- Function structure for gtrgm_same
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."gtrgm_same"("public"."gtrgm", "public"."gtrgm", internal);
CREATE FUNCTION "public"."gtrgm_same"("public"."gtrgm", "public"."gtrgm", internal)
  RETURNS "pg_catalog"."internal" AS '$libdir/pg_trgm', 'gtrgm_same'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."gtrgm_same"("public"."gtrgm", "public"."gtrgm", internal) OWNER TO "postgres";

-- ----------------------------
-- Function structure for gtrgm_union
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."gtrgm_union"(internal, internal);
CREATE FUNCTION "public"."gtrgm_union"(internal, internal)
  RETURNS "public"."gtrgm" AS '$libdir/pg_trgm', 'gtrgm_union'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."gtrgm_union"(internal, internal) OWNER TO "postgres";

-- ----------------------------
-- Function structure for halfvec
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."halfvec"("public"."halfvec", int4, bool);
CREATE FUNCTION "public"."halfvec"("public"."halfvec", int4, bool)
  RETURNS "public"."halfvec" AS '$libdir/vector', 'halfvec'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."halfvec"("public"."halfvec", int4, bool) OWNER TO "postgres";

-- ----------------------------
-- Function structure for halfvec_accum
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."halfvec_accum"(_float8, "public"."halfvec");
CREATE FUNCTION "public"."halfvec_accum"(_float8, "public"."halfvec")
  RETURNS "pg_catalog"."_float8" AS '$libdir/vector', 'halfvec_accum'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."halfvec_accum"(_float8, "public"."halfvec") OWNER TO "postgres";

-- ----------------------------
-- Function structure for halfvec_add
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."halfvec_add"("public"."halfvec", "public"."halfvec");
CREATE FUNCTION "public"."halfvec_add"("public"."halfvec", "public"."halfvec")
  RETURNS "public"."halfvec" AS '$libdir/vector', 'halfvec_add'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."halfvec_add"("public"."halfvec", "public"."halfvec") OWNER TO "postgres";

-- ----------------------------
-- Function structure for halfvec_avg
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."halfvec_avg"(_float8);
CREATE FUNCTION "public"."halfvec_avg"(_float8)
  RETURNS "public"."halfvec" AS '$libdir/vector', 'halfvec_avg'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."halfvec_avg"(_float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for halfvec_cmp
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."halfvec_cmp"("public"."halfvec", "public"."halfvec");
CREATE FUNCTION "public"."halfvec_cmp"("public"."halfvec", "public"."halfvec")
  RETURNS "pg_catalog"."int4" AS '$libdir/vector', 'halfvec_cmp'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."halfvec_cmp"("public"."halfvec", "public"."halfvec") OWNER TO "postgres";

-- ----------------------------
-- Function structure for halfvec_combine
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."halfvec_combine"(_float8, _float8);
CREATE FUNCTION "public"."halfvec_combine"(_float8, _float8)
  RETURNS "pg_catalog"."_float8" AS '$libdir/vector', 'vector_combine'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."halfvec_combine"(_float8, _float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for halfvec_concat
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."halfvec_concat"("public"."halfvec", "public"."halfvec");
CREATE FUNCTION "public"."halfvec_concat"("public"."halfvec", "public"."halfvec")
  RETURNS "public"."halfvec" AS '$libdir/vector', 'halfvec_concat'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."halfvec_concat"("public"."halfvec", "public"."halfvec") OWNER TO "postgres";

-- ----------------------------
-- Function structure for halfvec_eq
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."halfvec_eq"("public"."halfvec", "public"."halfvec");
CREATE FUNCTION "public"."halfvec_eq"("public"."halfvec", "public"."halfvec")
  RETURNS "pg_catalog"."bool" AS '$libdir/vector', 'halfvec_eq'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."halfvec_eq"("public"."halfvec", "public"."halfvec") OWNER TO "postgres";

-- ----------------------------
-- Function structure for halfvec_ge
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."halfvec_ge"("public"."halfvec", "public"."halfvec");
CREATE FUNCTION "public"."halfvec_ge"("public"."halfvec", "public"."halfvec")
  RETURNS "pg_catalog"."bool" AS '$libdir/vector', 'halfvec_ge'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."halfvec_ge"("public"."halfvec", "public"."halfvec") OWNER TO "postgres";

-- ----------------------------
-- Function structure for halfvec_gt
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."halfvec_gt"("public"."halfvec", "public"."halfvec");
CREATE FUNCTION "public"."halfvec_gt"("public"."halfvec", "public"."halfvec")
  RETURNS "pg_catalog"."bool" AS '$libdir/vector', 'halfvec_gt'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."halfvec_gt"("public"."halfvec", "public"."halfvec") OWNER TO "postgres";

-- ----------------------------
-- Function structure for halfvec_in
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."halfvec_in"(cstring, oid, int4);
CREATE FUNCTION "public"."halfvec_in"(cstring, oid, int4)
  RETURNS "public"."halfvec" AS '$libdir/vector', 'halfvec_in'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."halfvec_in"(cstring, oid, int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for halfvec_l2_squared_distance
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."halfvec_l2_squared_distance"("public"."halfvec", "public"."halfvec");
CREATE FUNCTION "public"."halfvec_l2_squared_distance"("public"."halfvec", "public"."halfvec")
  RETURNS "pg_catalog"."float8" AS '$libdir/vector', 'halfvec_l2_squared_distance'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."halfvec_l2_squared_distance"("public"."halfvec", "public"."halfvec") OWNER TO "postgres";

-- ----------------------------
-- Function structure for halfvec_le
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."halfvec_le"("public"."halfvec", "public"."halfvec");
CREATE FUNCTION "public"."halfvec_le"("public"."halfvec", "public"."halfvec")
  RETURNS "pg_catalog"."bool" AS '$libdir/vector', 'halfvec_le'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."halfvec_le"("public"."halfvec", "public"."halfvec") OWNER TO "postgres";

-- ----------------------------
-- Function structure for halfvec_lt
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."halfvec_lt"("public"."halfvec", "public"."halfvec");
CREATE FUNCTION "public"."halfvec_lt"("public"."halfvec", "public"."halfvec")
  RETURNS "pg_catalog"."bool" AS '$libdir/vector', 'halfvec_lt'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."halfvec_lt"("public"."halfvec", "public"."halfvec") OWNER TO "postgres";

-- ----------------------------
-- Function structure for halfvec_mul
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."halfvec_mul"("public"."halfvec", "public"."halfvec");
CREATE FUNCTION "public"."halfvec_mul"("public"."halfvec", "public"."halfvec")
  RETURNS "public"."halfvec" AS '$libdir/vector', 'halfvec_mul'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."halfvec_mul"("public"."halfvec", "public"."halfvec") OWNER TO "postgres";

-- ----------------------------
-- Function structure for halfvec_ne
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."halfvec_ne"("public"."halfvec", "public"."halfvec");
CREATE FUNCTION "public"."halfvec_ne"("public"."halfvec", "public"."halfvec")
  RETURNS "pg_catalog"."bool" AS '$libdir/vector', 'halfvec_ne'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."halfvec_ne"("public"."halfvec", "public"."halfvec") OWNER TO "postgres";

-- ----------------------------
-- Function structure for halfvec_negative_inner_product
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."halfvec_negative_inner_product"("public"."halfvec", "public"."halfvec");
CREATE FUNCTION "public"."halfvec_negative_inner_product"("public"."halfvec", "public"."halfvec")
  RETURNS "pg_catalog"."float8" AS '$libdir/vector', 'halfvec_negative_inner_product'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."halfvec_negative_inner_product"("public"."halfvec", "public"."halfvec") OWNER TO "postgres";

-- ----------------------------
-- Function structure for halfvec_out
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."halfvec_out"("public"."halfvec");
CREATE FUNCTION "public"."halfvec_out"("public"."halfvec")
  RETURNS "pg_catalog"."cstring" AS '$libdir/vector', 'halfvec_out'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."halfvec_out"("public"."halfvec") OWNER TO "postgres";

-- ----------------------------
-- Function structure for halfvec_recv
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."halfvec_recv"(internal, oid, int4);
CREATE FUNCTION "public"."halfvec_recv"(internal, oid, int4)
  RETURNS "public"."halfvec" AS '$libdir/vector', 'halfvec_recv'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."halfvec_recv"(internal, oid, int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for halfvec_send
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."halfvec_send"("public"."halfvec");
CREATE FUNCTION "public"."halfvec_send"("public"."halfvec")
  RETURNS "pg_catalog"."bytea" AS '$libdir/vector', 'halfvec_send'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."halfvec_send"("public"."halfvec") OWNER TO "postgres";

-- ----------------------------
-- Function structure for halfvec_spherical_distance
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."halfvec_spherical_distance"("public"."halfvec", "public"."halfvec");
CREATE FUNCTION "public"."halfvec_spherical_distance"("public"."halfvec", "public"."halfvec")
  RETURNS "pg_catalog"."float8" AS '$libdir/vector', 'halfvec_spherical_distance'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."halfvec_spherical_distance"("public"."halfvec", "public"."halfvec") OWNER TO "postgres";

-- ----------------------------
-- Function structure for halfvec_sub
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."halfvec_sub"("public"."halfvec", "public"."halfvec");
CREATE FUNCTION "public"."halfvec_sub"("public"."halfvec", "public"."halfvec")
  RETURNS "public"."halfvec" AS '$libdir/vector', 'halfvec_sub'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."halfvec_sub"("public"."halfvec", "public"."halfvec") OWNER TO "postgres";

-- ----------------------------
-- Function structure for halfvec_to_float4
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."halfvec_to_float4"("public"."halfvec", int4, bool);
CREATE FUNCTION "public"."halfvec_to_float4"("public"."halfvec", int4, bool)
  RETURNS "pg_catalog"."_float4" AS '$libdir/vector', 'halfvec_to_float4'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."halfvec_to_float4"("public"."halfvec", int4, bool) OWNER TO "postgres";

-- ----------------------------
-- Function structure for halfvec_to_sparsevec
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."halfvec_to_sparsevec"("public"."halfvec", int4, bool);
CREATE FUNCTION "public"."halfvec_to_sparsevec"("public"."halfvec", int4, bool)
  RETURNS "public"."sparsevec" AS '$libdir/vector', 'halfvec_to_sparsevec'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."halfvec_to_sparsevec"("public"."halfvec", int4, bool) OWNER TO "postgres";

-- ----------------------------
-- Function structure for halfvec_to_vector
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."halfvec_to_vector"("public"."halfvec", int4, bool);
CREATE FUNCTION "public"."halfvec_to_vector"("public"."halfvec", int4, bool)
  RETURNS "public"."vector" AS '$libdir/vector', 'halfvec_to_vector'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."halfvec_to_vector"("public"."halfvec", int4, bool) OWNER TO "postgres";

-- ----------------------------
-- Function structure for halfvec_typmod_in
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."halfvec_typmod_in"(_cstring);
CREATE FUNCTION "public"."halfvec_typmod_in"(_cstring)
  RETURNS "pg_catalog"."int4" AS '$libdir/vector', 'halfvec_typmod_in'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."halfvec_typmod_in"(_cstring) OWNER TO "postgres";

-- ----------------------------
-- Function structure for hamming_distance
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."hamming_distance"(bit, bit);
CREATE FUNCTION "public"."hamming_distance"(bit, bit)
  RETURNS "pg_catalog"."float8" AS '$libdir/vector', 'hamming_distance'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."hamming_distance"(bit, bit) OWNER TO "postgres";

-- ----------------------------
-- Function structure for hnsw_bit_support
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."hnsw_bit_support"(internal);
CREATE FUNCTION "public"."hnsw_bit_support"(internal)
  RETURNS "pg_catalog"."internal" AS '$libdir/vector', 'hnsw_bit_support'
  LANGUAGE c VOLATILE
  COST 1;
ALTER FUNCTION "public"."hnsw_bit_support"(internal) OWNER TO "postgres";

-- ----------------------------
-- Function structure for hnsw_halfvec_support
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."hnsw_halfvec_support"(internal);
CREATE FUNCTION "public"."hnsw_halfvec_support"(internal)
  RETURNS "pg_catalog"."internal" AS '$libdir/vector', 'hnsw_halfvec_support'
  LANGUAGE c VOLATILE
  COST 1;
ALTER FUNCTION "public"."hnsw_halfvec_support"(internal) OWNER TO "postgres";

-- ----------------------------
-- Function structure for hnsw_sparsevec_support
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."hnsw_sparsevec_support"(internal);
CREATE FUNCTION "public"."hnsw_sparsevec_support"(internal)
  RETURNS "pg_catalog"."internal" AS '$libdir/vector', 'hnsw_sparsevec_support'
  LANGUAGE c VOLATILE
  COST 1;
ALTER FUNCTION "public"."hnsw_sparsevec_support"(internal) OWNER TO "postgres";

-- ----------------------------
-- Function structure for hnswhandler
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."hnswhandler"(internal);
CREATE FUNCTION "public"."hnswhandler"(internal)
  RETURNS "pg_catalog"."index_am_handler" AS '$libdir/vector', 'hnswhandler'
  LANGUAGE c VOLATILE
  COST 1;
ALTER FUNCTION "public"."hnswhandler"(internal) OWNER TO "postgres";

-- ----------------------------
-- Function structure for inner_product
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."inner_product"("public"."vector", "public"."vector");
CREATE FUNCTION "public"."inner_product"("public"."vector", "public"."vector")
  RETURNS "pg_catalog"."float8" AS '$libdir/vector', 'inner_product'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."inner_product"("public"."vector", "public"."vector") OWNER TO "postgres";

-- ----------------------------
-- Function structure for inner_product
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."inner_product"("public"."halfvec", "public"."halfvec");
CREATE FUNCTION "public"."inner_product"("public"."halfvec", "public"."halfvec")
  RETURNS "pg_catalog"."float8" AS '$libdir/vector', 'halfvec_inner_product'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."inner_product"("public"."halfvec", "public"."halfvec") OWNER TO "postgres";

-- ----------------------------
-- Function structure for inner_product
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."inner_product"("public"."sparsevec", "public"."sparsevec");
CREATE FUNCTION "public"."inner_product"("public"."sparsevec", "public"."sparsevec")
  RETURNS "pg_catalog"."float8" AS '$libdir/vector', 'sparsevec_inner_product'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."inner_product"("public"."sparsevec", "public"."sparsevec") OWNER TO "postgres";

-- ----------------------------
-- Function structure for is_contained_2d
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."is_contained_2d"("public"."geometry", "public"."box2df");
CREATE FUNCTION "public"."is_contained_2d"("public"."geometry", "public"."box2df")
  RETURNS "pg_catalog"."bool" AS $BODY$SELECT $2 OPERATOR(public.~) $1;$BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."is_contained_2d"("public"."geometry", "public"."box2df") OWNER TO "postgres";

-- ----------------------------
-- Function structure for is_contained_2d
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."is_contained_2d"("public"."box2df", "public"."geometry");
CREATE FUNCTION "public"."is_contained_2d"("public"."box2df", "public"."geometry")
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'gserialized_within_box2df_geom_2d'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."is_contained_2d"("public"."box2df", "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for is_contained_2d
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."is_contained_2d"("public"."box2df", "public"."box2df");
CREATE FUNCTION "public"."is_contained_2d"("public"."box2df", "public"."box2df")
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'gserialized_contains_box2df_box2df_2d'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."is_contained_2d"("public"."box2df", "public"."box2df") OWNER TO "postgres";

-- ----------------------------
-- Function structure for ivfflat_bit_support
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."ivfflat_bit_support"(internal);
CREATE FUNCTION "public"."ivfflat_bit_support"(internal)
  RETURNS "pg_catalog"."internal" AS '$libdir/vector', 'ivfflat_bit_support'
  LANGUAGE c VOLATILE
  COST 1;
ALTER FUNCTION "public"."ivfflat_bit_support"(internal) OWNER TO "postgres";

-- ----------------------------
-- Function structure for ivfflat_halfvec_support
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."ivfflat_halfvec_support"(internal);
CREATE FUNCTION "public"."ivfflat_halfvec_support"(internal)
  RETURNS "pg_catalog"."internal" AS '$libdir/vector', 'ivfflat_halfvec_support'
  LANGUAGE c VOLATILE
  COST 1;
ALTER FUNCTION "public"."ivfflat_halfvec_support"(internal) OWNER TO "postgres";

-- ----------------------------
-- Function structure for ivfflathandler
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."ivfflathandler"(internal);
CREATE FUNCTION "public"."ivfflathandler"(internal)
  RETURNS "pg_catalog"."index_am_handler" AS '$libdir/vector', 'ivfflathandler'
  LANGUAGE c VOLATILE
  COST 1;
ALTER FUNCTION "public"."ivfflathandler"(internal) OWNER TO "postgres";

-- ----------------------------
-- Function structure for jaccard_distance
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."jaccard_distance"(bit, bit);
CREATE FUNCTION "public"."jaccard_distance"(bit, bit)
  RETURNS "pg_catalog"."float8" AS '$libdir/vector', 'jaccard_distance'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."jaccard_distance"(bit, bit) OWNER TO "postgres";

-- ----------------------------
-- Function structure for json
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."json"("public"."geometry");
CREATE FUNCTION "public"."json"("public"."geometry")
  RETURNS "pg_catalog"."json" AS '$libdir/postgis-3', 'geometry_to_json'
  LANGUAGE c IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."json"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for jsonb
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."jsonb"("public"."geometry");
CREATE FUNCTION "public"."jsonb"("public"."geometry")
  RETURNS "pg_catalog"."jsonb" AS '$libdir/postgis-3', 'geometry_to_jsonb'
  LANGUAGE c IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."jsonb"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for l1_distance
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."l1_distance"("public"."sparsevec", "public"."sparsevec");
CREATE FUNCTION "public"."l1_distance"("public"."sparsevec", "public"."sparsevec")
  RETURNS "pg_catalog"."float8" AS '$libdir/vector', 'sparsevec_l1_distance'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."l1_distance"("public"."sparsevec", "public"."sparsevec") OWNER TO "postgres";

-- ----------------------------
-- Function structure for l1_distance
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."l1_distance"("public"."vector", "public"."vector");
CREATE FUNCTION "public"."l1_distance"("public"."vector", "public"."vector")
  RETURNS "pg_catalog"."float8" AS '$libdir/vector', 'l1_distance'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."l1_distance"("public"."vector", "public"."vector") OWNER TO "postgres";

-- ----------------------------
-- Function structure for l1_distance
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."l1_distance"("public"."halfvec", "public"."halfvec");
CREATE FUNCTION "public"."l1_distance"("public"."halfvec", "public"."halfvec")
  RETURNS "pg_catalog"."float8" AS '$libdir/vector', 'halfvec_l1_distance'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."l1_distance"("public"."halfvec", "public"."halfvec") OWNER TO "postgres";

-- ----------------------------
-- Function structure for l2_distance
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."l2_distance"("public"."vector", "public"."vector");
CREATE FUNCTION "public"."l2_distance"("public"."vector", "public"."vector")
  RETURNS "pg_catalog"."float8" AS '$libdir/vector', 'l2_distance'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."l2_distance"("public"."vector", "public"."vector") OWNER TO "postgres";

-- ----------------------------
-- Function structure for l2_distance
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."l2_distance"("public"."halfvec", "public"."halfvec");
CREATE FUNCTION "public"."l2_distance"("public"."halfvec", "public"."halfvec")
  RETURNS "pg_catalog"."float8" AS '$libdir/vector', 'halfvec_l2_distance'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."l2_distance"("public"."halfvec", "public"."halfvec") OWNER TO "postgres";

-- ----------------------------
-- Function structure for l2_distance
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."l2_distance"("public"."sparsevec", "public"."sparsevec");
CREATE FUNCTION "public"."l2_distance"("public"."sparsevec", "public"."sparsevec")
  RETURNS "pg_catalog"."float8" AS '$libdir/vector', 'sparsevec_l2_distance'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."l2_distance"("public"."sparsevec", "public"."sparsevec") OWNER TO "postgres";

-- ----------------------------
-- Function structure for l2_norm
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."l2_norm"("public"."halfvec");
CREATE FUNCTION "public"."l2_norm"("public"."halfvec")
  RETURNS "pg_catalog"."float8" AS '$libdir/vector', 'halfvec_l2_norm'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."l2_norm"("public"."halfvec") OWNER TO "postgres";

-- ----------------------------
-- Function structure for l2_norm
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."l2_norm"("public"."sparsevec");
CREATE FUNCTION "public"."l2_norm"("public"."sparsevec")
  RETURNS "pg_catalog"."float8" AS '$libdir/vector', 'sparsevec_l2_norm'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."l2_norm"("public"."sparsevec") OWNER TO "postgres";

-- ----------------------------
-- Function structure for l2_normalize
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."l2_normalize"("public"."vector");
CREATE FUNCTION "public"."l2_normalize"("public"."vector")
  RETURNS "public"."vector" AS '$libdir/vector', 'l2_normalize'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."l2_normalize"("public"."vector") OWNER TO "postgres";

-- ----------------------------
-- Function structure for l2_normalize
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."l2_normalize"("public"."sparsevec");
CREATE FUNCTION "public"."l2_normalize"("public"."sparsevec")
  RETURNS "public"."sparsevec" AS '$libdir/vector', 'sparsevec_l2_normalize'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."l2_normalize"("public"."sparsevec") OWNER TO "postgres";

-- ----------------------------
-- Function structure for l2_normalize
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."l2_normalize"("public"."halfvec");
CREATE FUNCTION "public"."l2_normalize"("public"."halfvec")
  RETURNS "public"."halfvec" AS '$libdir/vector', 'halfvec_l2_normalize'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."l2_normalize"("public"."halfvec") OWNER TO "postgres";

-- ----------------------------
-- Function structure for levenshtein
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."levenshtein"(text, text);
CREATE FUNCTION "public"."levenshtein"(text, text)
  RETURNS "pg_catalog"."int4" AS '$libdir/fuzzystrmatch', 'levenshtein'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."levenshtein"(text, text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for levenshtein
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."levenshtein"(text, text, int4, int4, int4);
CREATE FUNCTION "public"."levenshtein"(text, text, int4, int4, int4)
  RETURNS "pg_catalog"."int4" AS '$libdir/fuzzystrmatch', 'levenshtein_with_costs'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."levenshtein"(text, text, int4, int4, int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for levenshtein_less_equal
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."levenshtein_less_equal"(text, text, int4);
CREATE FUNCTION "public"."levenshtein_less_equal"(text, text, int4)
  RETURNS "pg_catalog"."int4" AS '$libdir/fuzzystrmatch', 'levenshtein_less_equal'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."levenshtein_less_equal"(text, text, int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for levenshtein_less_equal
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."levenshtein_less_equal"(text, text, int4, int4, int4, int4);
CREATE FUNCTION "public"."levenshtein_less_equal"(text, text, int4, int4, int4, int4)
  RETURNS "pg_catalog"."int4" AS '$libdir/fuzzystrmatch', 'levenshtein_less_equal_with_costs'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."levenshtein_less_equal"(text, text, int4, int4, int4, int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for metaphone
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."metaphone"(text, int4);
CREATE FUNCTION "public"."metaphone"(text, int4)
  RETURNS "pg_catalog"."text" AS '$libdir/fuzzystrmatch', 'metaphone'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."metaphone"(text, int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for overlaps_2d
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."overlaps_2d"("public"."geometry", "public"."box2df");
CREATE FUNCTION "public"."overlaps_2d"("public"."geometry", "public"."box2df")
  RETURNS "pg_catalog"."bool" AS $BODY$SELECT $2 OPERATOR(public.&&) $1;$BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."overlaps_2d"("public"."geometry", "public"."box2df") OWNER TO "postgres";

-- ----------------------------
-- Function structure for overlaps_2d
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."overlaps_2d"("public"."box2df", "public"."box2df");
CREATE FUNCTION "public"."overlaps_2d"("public"."box2df", "public"."box2df")
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'gserialized_contains_box2df_box2df_2d'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."overlaps_2d"("public"."box2df", "public"."box2df") OWNER TO "postgres";

-- ----------------------------
-- Function structure for overlaps_2d
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."overlaps_2d"("public"."box2df", "public"."geometry");
CREATE FUNCTION "public"."overlaps_2d"("public"."box2df", "public"."geometry")
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'gserialized_overlaps_box2df_geom_2d'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."overlaps_2d"("public"."box2df", "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for overlaps_geog
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."overlaps_geog"("public"."gidx", "public"."gidx");
CREATE FUNCTION "public"."overlaps_geog"("public"."gidx", "public"."gidx")
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'gserialized_gidx_gidx_overlaps'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."overlaps_geog"("public"."gidx", "public"."gidx") OWNER TO "postgres";

-- ----------------------------
-- Function structure for overlaps_geog
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."overlaps_geog"("public"."gidx", "public"."geography");
CREATE FUNCTION "public"."overlaps_geog"("public"."gidx", "public"."geography")
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'gserialized_gidx_geog_overlaps'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."overlaps_geog"("public"."gidx", "public"."geography") OWNER TO "postgres";

-- ----------------------------
-- Function structure for overlaps_geog
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."overlaps_geog"("public"."geography", "public"."gidx");
CREATE FUNCTION "public"."overlaps_geog"("public"."geography", "public"."gidx")
  RETURNS "pg_catalog"."bool" AS $BODY$SELECT $2 OPERATOR(public.&&) $1;$BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 100;
ALTER FUNCTION "public"."overlaps_geog"("public"."geography", "public"."gidx") OWNER TO "postgres";

-- ----------------------------
-- Function structure for overlaps_nd
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."overlaps_nd"("public"."gidx", "public"."geometry");
CREATE FUNCTION "public"."overlaps_nd"("public"."gidx", "public"."geometry")
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'gserialized_gidx_geom_overlaps'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."overlaps_nd"("public"."gidx", "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for overlaps_nd
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."overlaps_nd"("public"."geometry", "public"."gidx");
CREATE FUNCTION "public"."overlaps_nd"("public"."geometry", "public"."gidx")
  RETURNS "pg_catalog"."bool" AS $BODY$SELECT $2 OPERATOR(public.&&&) $1;$BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."overlaps_nd"("public"."geometry", "public"."gidx") OWNER TO "postgres";

-- ----------------------------
-- Function structure for overlaps_nd
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."overlaps_nd"("public"."gidx", "public"."gidx");
CREATE FUNCTION "public"."overlaps_nd"("public"."gidx", "public"."gidx")
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'gserialized_gidx_gidx_overlaps'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."overlaps_nd"("public"."gidx", "public"."gidx") OWNER TO "postgres";

-- ----------------------------
-- Function structure for path
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."path"("public"."geometry");
CREATE FUNCTION "public"."path"("public"."geometry")
  RETURNS "pg_catalog"."path" AS '$libdir/postgis-3', 'geometry_to_path'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."path"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for pgis_asflatgeobuf_finalfn
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."pgis_asflatgeobuf_finalfn"(internal);
CREATE FUNCTION "public"."pgis_asflatgeobuf_finalfn"(internal)
  RETURNS "pg_catalog"."bytea" AS '$libdir/postgis-3', 'pgis_asflatgeobuf_finalfn'
  LANGUAGE c IMMUTABLE
  COST 250;
ALTER FUNCTION "public"."pgis_asflatgeobuf_finalfn"(internal) OWNER TO "postgres";

-- ----------------------------
-- Function structure for pgis_asflatgeobuf_transfn
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."pgis_asflatgeobuf_transfn"(internal, anyelement, bool, text);
CREATE FUNCTION "public"."pgis_asflatgeobuf_transfn"(internal, anyelement, bool, text)
  RETURNS "pg_catalog"."internal" AS '$libdir/postgis-3', 'pgis_asflatgeobuf_transfn'
  LANGUAGE c IMMUTABLE
  COST 50;
ALTER FUNCTION "public"."pgis_asflatgeobuf_transfn"(internal, anyelement, bool, text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for pgis_asflatgeobuf_transfn
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."pgis_asflatgeobuf_transfn"(internal, anyelement, bool);
CREATE FUNCTION "public"."pgis_asflatgeobuf_transfn"(internal, anyelement, bool)
  RETURNS "pg_catalog"."internal" AS '$libdir/postgis-3', 'pgis_asflatgeobuf_transfn'
  LANGUAGE c IMMUTABLE
  COST 50;
ALTER FUNCTION "public"."pgis_asflatgeobuf_transfn"(internal, anyelement, bool) OWNER TO "postgres";

-- ----------------------------
-- Function structure for pgis_asflatgeobuf_transfn
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."pgis_asflatgeobuf_transfn"(internal, anyelement);
CREATE FUNCTION "public"."pgis_asflatgeobuf_transfn"(internal, anyelement)
  RETURNS "pg_catalog"."internal" AS '$libdir/postgis-3', 'pgis_asflatgeobuf_transfn'
  LANGUAGE c IMMUTABLE
  COST 50;
ALTER FUNCTION "public"."pgis_asflatgeobuf_transfn"(internal, anyelement) OWNER TO "postgres";

-- ----------------------------
-- Function structure for pgis_asgeobuf_finalfn
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."pgis_asgeobuf_finalfn"(internal);
CREATE FUNCTION "public"."pgis_asgeobuf_finalfn"(internal)
  RETURNS "pg_catalog"."bytea" AS '$libdir/postgis-3', 'pgis_asgeobuf_finalfn'
  LANGUAGE c IMMUTABLE
  COST 250;
ALTER FUNCTION "public"."pgis_asgeobuf_finalfn"(internal) OWNER TO "postgres";

-- ----------------------------
-- Function structure for pgis_asgeobuf_transfn
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."pgis_asgeobuf_transfn"(internal, anyelement, text);
CREATE FUNCTION "public"."pgis_asgeobuf_transfn"(internal, anyelement, text)
  RETURNS "pg_catalog"."internal" AS '$libdir/postgis-3', 'pgis_asgeobuf_transfn'
  LANGUAGE c IMMUTABLE
  COST 50;
ALTER FUNCTION "public"."pgis_asgeobuf_transfn"(internal, anyelement, text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for pgis_asgeobuf_transfn
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."pgis_asgeobuf_transfn"(internal, anyelement);
CREATE FUNCTION "public"."pgis_asgeobuf_transfn"(internal, anyelement)
  RETURNS "pg_catalog"."internal" AS '$libdir/postgis-3', 'pgis_asgeobuf_transfn'
  LANGUAGE c IMMUTABLE
  COST 50;
ALTER FUNCTION "public"."pgis_asgeobuf_transfn"(internal, anyelement) OWNER TO "postgres";

-- ----------------------------
-- Function structure for pgis_asmvt_combinefn
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."pgis_asmvt_combinefn"(internal, internal);
CREATE FUNCTION "public"."pgis_asmvt_combinefn"(internal, internal)
  RETURNS "pg_catalog"."internal" AS '$libdir/postgis-3', 'pgis_asmvt_combinefn'
  LANGUAGE c IMMUTABLE
  COST 250;
ALTER FUNCTION "public"."pgis_asmvt_combinefn"(internal, internal) OWNER TO "postgres";

-- ----------------------------
-- Function structure for pgis_asmvt_deserialfn
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."pgis_asmvt_deserialfn"(bytea, internal);
CREATE FUNCTION "public"."pgis_asmvt_deserialfn"(bytea, internal)
  RETURNS "pg_catalog"."internal" AS '$libdir/postgis-3', 'pgis_asmvt_deserialfn'
  LANGUAGE c IMMUTABLE
  COST 250;
ALTER FUNCTION "public"."pgis_asmvt_deserialfn"(bytea, internal) OWNER TO "postgres";

-- ----------------------------
-- Function structure for pgis_asmvt_finalfn
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."pgis_asmvt_finalfn"(internal);
CREATE FUNCTION "public"."pgis_asmvt_finalfn"(internal)
  RETURNS "pg_catalog"."bytea" AS '$libdir/postgis-3', 'pgis_asmvt_finalfn'
  LANGUAGE c IMMUTABLE
  COST 250;
ALTER FUNCTION "public"."pgis_asmvt_finalfn"(internal) OWNER TO "postgres";

-- ----------------------------
-- Function structure for pgis_asmvt_serialfn
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."pgis_asmvt_serialfn"(internal);
CREATE FUNCTION "public"."pgis_asmvt_serialfn"(internal)
  RETURNS "pg_catalog"."bytea" AS '$libdir/postgis-3', 'pgis_asmvt_serialfn'
  LANGUAGE c IMMUTABLE
  COST 250;
ALTER FUNCTION "public"."pgis_asmvt_serialfn"(internal) OWNER TO "postgres";

-- ----------------------------
-- Function structure for pgis_asmvt_transfn
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."pgis_asmvt_transfn"(internal, anyelement);
CREATE FUNCTION "public"."pgis_asmvt_transfn"(internal, anyelement)
  RETURNS "pg_catalog"."internal" AS '$libdir/postgis-3', 'pgis_asmvt_transfn'
  LANGUAGE c IMMUTABLE
  COST 250;
ALTER FUNCTION "public"."pgis_asmvt_transfn"(internal, anyelement) OWNER TO "postgres";

-- ----------------------------
-- Function structure for pgis_asmvt_transfn
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."pgis_asmvt_transfn"(internal, anyelement, text);
CREATE FUNCTION "public"."pgis_asmvt_transfn"(internal, anyelement, text)
  RETURNS "pg_catalog"."internal" AS '$libdir/postgis-3', 'pgis_asmvt_transfn'
  LANGUAGE c IMMUTABLE
  COST 250;
ALTER FUNCTION "public"."pgis_asmvt_transfn"(internal, anyelement, text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for pgis_asmvt_transfn
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."pgis_asmvt_transfn"(internal, anyelement, text, int4);
CREATE FUNCTION "public"."pgis_asmvt_transfn"(internal, anyelement, text, int4)
  RETURNS "pg_catalog"."internal" AS '$libdir/postgis-3', 'pgis_asmvt_transfn'
  LANGUAGE c IMMUTABLE
  COST 250;
ALTER FUNCTION "public"."pgis_asmvt_transfn"(internal, anyelement, text, int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for pgis_asmvt_transfn
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."pgis_asmvt_transfn"(internal, anyelement, text, int4, text);
CREATE FUNCTION "public"."pgis_asmvt_transfn"(internal, anyelement, text, int4, text)
  RETURNS "pg_catalog"."internal" AS '$libdir/postgis-3', 'pgis_asmvt_transfn'
  LANGUAGE c IMMUTABLE
  COST 250;
ALTER FUNCTION "public"."pgis_asmvt_transfn"(internal, anyelement, text, int4, text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for pgis_asmvt_transfn
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."pgis_asmvt_transfn"(internal, anyelement, text, int4, text, text);
CREATE FUNCTION "public"."pgis_asmvt_transfn"(internal, anyelement, text, int4, text, text)
  RETURNS "pg_catalog"."internal" AS '$libdir/postgis-3', 'pgis_asmvt_transfn'
  LANGUAGE c IMMUTABLE
  COST 250;
ALTER FUNCTION "public"."pgis_asmvt_transfn"(internal, anyelement, text, int4, text, text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for pgis_geometry_accum_transfn
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."pgis_geometry_accum_transfn"(internal, "public"."geometry", float8);
CREATE FUNCTION "public"."pgis_geometry_accum_transfn"(internal, "public"."geometry", float8)
  RETURNS "pg_catalog"."internal" AS '$libdir/postgis-3', 'pgis_geometry_accum_transfn'
  LANGUAGE c VOLATILE
  COST 50;
ALTER FUNCTION "public"."pgis_geometry_accum_transfn"(internal, "public"."geometry", float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for pgis_geometry_accum_transfn
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."pgis_geometry_accum_transfn"(internal, "public"."geometry", float8, int4);
CREATE FUNCTION "public"."pgis_geometry_accum_transfn"(internal, "public"."geometry", float8, int4)
  RETURNS "pg_catalog"."internal" AS '$libdir/postgis-3', 'pgis_geometry_accum_transfn'
  LANGUAGE c VOLATILE
  COST 50;
ALTER FUNCTION "public"."pgis_geometry_accum_transfn"(internal, "public"."geometry", float8, int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for pgis_geometry_accum_transfn
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."pgis_geometry_accum_transfn"(internal, "public"."geometry");
CREATE FUNCTION "public"."pgis_geometry_accum_transfn"(internal, "public"."geometry")
  RETURNS "pg_catalog"."internal" AS '$libdir/postgis-3', 'pgis_geometry_accum_transfn'
  LANGUAGE c VOLATILE
  COST 50;
ALTER FUNCTION "public"."pgis_geometry_accum_transfn"(internal, "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for pgis_geometry_clusterintersecting_finalfn
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."pgis_geometry_clusterintersecting_finalfn"(internal);
CREATE FUNCTION "public"."pgis_geometry_clusterintersecting_finalfn"(internal)
  RETURNS "public"."_geometry" AS '$libdir/postgis-3', 'pgis_geometry_clusterintersecting_finalfn'
  LANGUAGE c VOLATILE
  COST 250;
ALTER FUNCTION "public"."pgis_geometry_clusterintersecting_finalfn"(internal) OWNER TO "postgres";

-- ----------------------------
-- Function structure for pgis_geometry_clusterwithin_finalfn
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."pgis_geometry_clusterwithin_finalfn"(internal);
CREATE FUNCTION "public"."pgis_geometry_clusterwithin_finalfn"(internal)
  RETURNS "public"."_geometry" AS '$libdir/postgis-3', 'pgis_geometry_clusterwithin_finalfn'
  LANGUAGE c VOLATILE
  COST 250;
ALTER FUNCTION "public"."pgis_geometry_clusterwithin_finalfn"(internal) OWNER TO "postgres";

-- ----------------------------
-- Function structure for pgis_geometry_collect_finalfn
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."pgis_geometry_collect_finalfn"(internal);
CREATE FUNCTION "public"."pgis_geometry_collect_finalfn"(internal)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'pgis_geometry_collect_finalfn'
  LANGUAGE c VOLATILE
  COST 250;
ALTER FUNCTION "public"."pgis_geometry_collect_finalfn"(internal) OWNER TO "postgres";

-- ----------------------------
-- Function structure for pgis_geometry_coverageunion_finalfn
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."pgis_geometry_coverageunion_finalfn"(internal);
CREATE FUNCTION "public"."pgis_geometry_coverageunion_finalfn"(internal)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'pgis_geometry_coverageunion_finalfn'
  LANGUAGE c VOLATILE
  COST 250;
ALTER FUNCTION "public"."pgis_geometry_coverageunion_finalfn"(internal) OWNER TO "postgres";

-- ----------------------------
-- Function structure for pgis_geometry_makeline_finalfn
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."pgis_geometry_makeline_finalfn"(internal);
CREATE FUNCTION "public"."pgis_geometry_makeline_finalfn"(internal)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'pgis_geometry_makeline_finalfn'
  LANGUAGE c VOLATILE
  COST 250;
ALTER FUNCTION "public"."pgis_geometry_makeline_finalfn"(internal) OWNER TO "postgres";

-- ----------------------------
-- Function structure for pgis_geometry_polygonize_finalfn
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."pgis_geometry_polygonize_finalfn"(internal);
CREATE FUNCTION "public"."pgis_geometry_polygonize_finalfn"(internal)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'pgis_geometry_polygonize_finalfn'
  LANGUAGE c VOLATILE
  COST 250;
ALTER FUNCTION "public"."pgis_geometry_polygonize_finalfn"(internal) OWNER TO "postgres";

-- ----------------------------
-- Function structure for pgis_geometry_union_parallel_combinefn
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."pgis_geometry_union_parallel_combinefn"(internal, internal);
CREATE FUNCTION "public"."pgis_geometry_union_parallel_combinefn"(internal, internal)
  RETURNS "pg_catalog"."internal" AS '$libdir/postgis-3', 'pgis_geometry_union_parallel_combinefn'
  LANGUAGE c IMMUTABLE
  COST 1;
ALTER FUNCTION "public"."pgis_geometry_union_parallel_combinefn"(internal, internal) OWNER TO "postgres";

-- ----------------------------
-- Function structure for pgis_geometry_union_parallel_deserialfn
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."pgis_geometry_union_parallel_deserialfn"(bytea, internal);
CREATE FUNCTION "public"."pgis_geometry_union_parallel_deserialfn"(bytea, internal)
  RETURNS "pg_catalog"."internal" AS '$libdir/postgis-3', 'pgis_geometry_union_parallel_deserialfn'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."pgis_geometry_union_parallel_deserialfn"(bytea, internal) OWNER TO "postgres";

-- ----------------------------
-- Function structure for pgis_geometry_union_parallel_finalfn
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."pgis_geometry_union_parallel_finalfn"(internal);
CREATE FUNCTION "public"."pgis_geometry_union_parallel_finalfn"(internal)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'pgis_geometry_union_parallel_finalfn'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."pgis_geometry_union_parallel_finalfn"(internal) OWNER TO "postgres";

-- ----------------------------
-- Function structure for pgis_geometry_union_parallel_serialfn
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."pgis_geometry_union_parallel_serialfn"(internal);
CREATE FUNCTION "public"."pgis_geometry_union_parallel_serialfn"(internal)
  RETURNS "pg_catalog"."bytea" AS '$libdir/postgis-3', 'pgis_geometry_union_parallel_serialfn'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."pgis_geometry_union_parallel_serialfn"(internal) OWNER TO "postgres";

-- ----------------------------
-- Function structure for pgis_geometry_union_parallel_transfn
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."pgis_geometry_union_parallel_transfn"(internal, "public"."geometry", float8);
CREATE FUNCTION "public"."pgis_geometry_union_parallel_transfn"(internal, "public"."geometry", float8)
  RETURNS "pg_catalog"."internal" AS '$libdir/postgis-3', 'pgis_geometry_union_parallel_transfn'
  LANGUAGE c IMMUTABLE
  COST 50;
ALTER FUNCTION "public"."pgis_geometry_union_parallel_transfn"(internal, "public"."geometry", float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for pgis_geometry_union_parallel_transfn
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."pgis_geometry_union_parallel_transfn"(internal, "public"."geometry");
CREATE FUNCTION "public"."pgis_geometry_union_parallel_transfn"(internal, "public"."geometry")
  RETURNS "pg_catalog"."internal" AS '$libdir/postgis-3', 'pgis_geometry_union_parallel_transfn'
  LANGUAGE c IMMUTABLE
  COST 1;
ALTER FUNCTION "public"."pgis_geometry_union_parallel_transfn"(internal, "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for point
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."point"("public"."geometry");
CREATE FUNCTION "public"."point"("public"."geometry")
  RETURNS "pg_catalog"."point" AS '$libdir/postgis-3', 'geometry_to_point'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."point"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for polygon
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."polygon"("public"."geometry");
CREATE FUNCTION "public"."polygon"("public"."geometry")
  RETURNS "pg_catalog"."polygon" AS '$libdir/postgis-3', 'geometry_to_polygon'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."polygon"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for populate_geometry_columns
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."populate_geometry_columns"("use_typmod" bool);
CREATE FUNCTION "public"."populate_geometry_columns"("use_typmod" bool=true)
  RETURNS "pg_catalog"."text" AS $BODY$
DECLARE
	inserted	integer;
	oldcount	integer;
	probed	  integer;
	stale	   integer;
	gcs		 RECORD;
	gc		  RECORD;
	gsrid	   integer;
	gndims	  integer;
	gtype	   text;
	query	   text;
	gc_is_valid boolean;

BEGIN
	SELECT count(*) INTO oldcount FROM public.geometry_columns;
	inserted := 0;

	-- Count the number of geometry columns in all tables and views
	SELECT count(DISTINCT c.oid) INTO probed
	FROM pg_class c,
		 pg_attribute a,
		 pg_type t,
		 pg_namespace n
	WHERE c.relkind IN('r','v','f', 'p')
		AND t.typname = 'geometry'
		AND a.attisdropped = false
		AND a.atttypid = t.oid
		AND a.attrelid = c.oid
		AND c.relnamespace = n.oid
		AND n.nspname NOT ILIKE 'pg_temp%' AND c.relname != 'raster_columns' ;

	-- Iterate through all non-dropped geometry columns
	RAISE DEBUG 'Processing Tables.....';

	FOR gcs IN
	SELECT DISTINCT ON (c.oid) c.oid, n.nspname, c.relname
		FROM pg_class c,
			 pg_attribute a,
			 pg_type t,
			 pg_namespace n
		WHERE c.relkind IN( 'r', 'f', 'p')
		AND t.typname = 'geometry'
		AND a.attisdropped = false
		AND a.atttypid = t.oid
		AND a.attrelid = c.oid
		AND c.relnamespace = n.oid
		AND n.nspname NOT ILIKE 'pg_temp%' AND c.relname != 'raster_columns'
	LOOP

		inserted := inserted + public.populate_geometry_columns(gcs.oid, use_typmod);
	END LOOP;

	IF oldcount > inserted THEN
		stale = oldcount-inserted;
	ELSE
		stale = 0;
	END IF;

	RETURN 'probed:' ||probed|| ' inserted:'||inserted;
END

$BODY$
  LANGUAGE plpgsql VOLATILE
  COST 100;
ALTER FUNCTION "public"."populate_geometry_columns"("use_typmod" bool) OWNER TO "postgres";

-- ----------------------------
-- Function structure for populate_geometry_columns
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."populate_geometry_columns"("tbl_oid" oid, "use_typmod" bool);
CREATE FUNCTION "public"."populate_geometry_columns"("tbl_oid" oid, "use_typmod" bool=true)
  RETURNS "pg_catalog"."int4" AS $BODY$
DECLARE
	gcs		 RECORD;
	gc		  RECORD;
	gc_old	  RECORD;
	gsrid	   integer;
	gndims	  integer;
	gtype	   text;
	query	   text;
	gc_is_valid boolean;
	inserted	integer;
	constraint_successful boolean := false;

BEGIN
	inserted := 0;

	-- Iterate through all geometry columns in this table
	FOR gcs IN
	SELECT n.nspname, c.relname, a.attname, c.relkind
		FROM pg_class c,
			 pg_attribute a,
			 pg_type t,
			 pg_namespace n
		WHERE c.relkind IN('r', 'f', 'p')
		AND t.typname = 'geometry'
		AND a.attisdropped = false
		AND a.atttypid = t.oid
		AND a.attrelid = c.oid
		AND c.relnamespace = n.oid
		AND n.nspname NOT ILIKE 'pg_temp%'
		AND c.oid = tbl_oid
	LOOP

		RAISE DEBUG 'Processing column %.%.%', gcs.nspname, gcs.relname, gcs.attname;

		gc_is_valid := true;
		-- Find the srid, coord_dimension, and type of current geometry
		-- in geometry_columns -- which is now a view

		SELECT type, srid, coord_dimension, gcs.relkind INTO gc_old
			FROM geometry_columns
			WHERE f_table_schema = gcs.nspname AND f_table_name = gcs.relname AND f_geometry_column = gcs.attname;

		IF upper(gc_old.type) = 'GEOMETRY' THEN
		-- This is an unconstrained geometry we need to do something
		-- We need to figure out what to set the type by inspecting the data
			EXECUTE 'SELECT public.ST_srid(' || quote_ident(gcs.attname) || ') As srid, public.GeometryType(' || quote_ident(gcs.attname) || ') As type, public.ST_NDims(' || quote_ident(gcs.attname) || ') As dims ' ||
					 ' FROM ONLY ' || quote_ident(gcs.nspname) || '.' || quote_ident(gcs.relname) ||
					 ' WHERE ' || quote_ident(gcs.attname) || ' IS NOT NULL LIMIT 1;'
				INTO gc;
			IF gc IS NULL THEN -- there is no data so we can not determine geometry type
				RAISE WARNING 'No data in table %.%, so no information to determine geometry type and srid', gcs.nspname, gcs.relname;
				RETURN 0;
			END IF;
			gsrid := gc.srid; gtype := gc.type; gndims := gc.dims;

			IF use_typmod THEN
				BEGIN
					EXECUTE 'ALTER TABLE ' || quote_ident(gcs.nspname) || '.' || quote_ident(gcs.relname) || ' ALTER COLUMN ' || quote_ident(gcs.attname) ||
						' TYPE geometry(' || postgis_type_name(gtype, gndims, true) || ', ' || gsrid::text  || ') ';
					inserted := inserted + 1;
				EXCEPTION
						WHEN invalid_parameter_value OR feature_not_supported THEN
						RAISE WARNING 'Could not convert ''%'' in ''%.%'' to use typmod with srid %, type %: %', quote_ident(gcs.attname), quote_ident(gcs.nspname), quote_ident(gcs.relname), gsrid, postgis_type_name(gtype, gndims, true), SQLERRM;
							gc_is_valid := false;
				END;

			ELSE
				-- Try to apply srid check to column
				constraint_successful = false;
				IF (gsrid > 0 AND postgis_constraint_srid(gcs.nspname, gcs.relname,gcs.attname) IS NULL ) THEN
					BEGIN
						EXECUTE 'ALTER TABLE ONLY ' || quote_ident(gcs.nspname) || '.' || quote_ident(gcs.relname) ||
								 ' ADD CONSTRAINT ' || quote_ident('enforce_srid_' || gcs.attname) ||
								 ' CHECK (ST_srid(' || quote_ident(gcs.attname) || ') = ' || gsrid || ')';
						constraint_successful := true;
					EXCEPTION
						WHEN check_violation THEN
							RAISE WARNING 'Not inserting ''%'' in ''%.%'' into geometry_columns: could not apply constraint CHECK (st_srid(%) = %)', quote_ident(gcs.attname), quote_ident(gcs.nspname), quote_ident(gcs.relname), quote_ident(gcs.attname), gsrid;
							gc_is_valid := false;
					END;
				END IF;

				-- Try to apply ndims check to column
				IF (gndims IS NOT NULL AND postgis_constraint_dims(gcs.nspname, gcs.relname,gcs.attname) IS NULL ) THEN
					BEGIN
						EXECUTE 'ALTER TABLE ONLY ' || quote_ident(gcs.nspname) || '.' || quote_ident(gcs.relname) || '
								 ADD CONSTRAINT ' || quote_ident('enforce_dims_' || gcs.attname) || '
								 CHECK (st_ndims(' || quote_ident(gcs.attname) || ') = '||gndims||')';
						constraint_successful := true;
					EXCEPTION
						WHEN check_violation THEN
							RAISE WARNING 'Not inserting ''%'' in ''%.%'' into geometry_columns: could not apply constraint CHECK (st_ndims(%) = %)', quote_ident(gcs.attname), quote_ident(gcs.nspname), quote_ident(gcs.relname), quote_ident(gcs.attname), gndims;
							gc_is_valid := false;
					END;
				END IF;

				-- Try to apply geometrytype check to column
				IF (gtype IS NOT NULL AND postgis_constraint_type(gcs.nspname, gcs.relname,gcs.attname) IS NULL ) THEN
					BEGIN
						EXECUTE 'ALTER TABLE ONLY ' || quote_ident(gcs.nspname) || '.' || quote_ident(gcs.relname) || '
						ADD CONSTRAINT ' || quote_ident('enforce_geotype_' || gcs.attname) || '
						CHECK (geometrytype(' || quote_ident(gcs.attname) || ') = ' || quote_literal(gtype) || ')';
						constraint_successful := true;
					EXCEPTION
						WHEN check_violation THEN
							-- No geometry check can be applied. This column contains a number of geometry types.
							RAISE WARNING 'Could not add geometry type check (%) to table column: %.%.%', gtype, quote_ident(gcs.nspname),quote_ident(gcs.relname),quote_ident(gcs.attname);
					END;
				END IF;
				 --only count if we were successful in applying at least one constraint
				IF constraint_successful THEN
					inserted := inserted + 1;
				END IF;
			END IF;
		END IF;

	END LOOP;

	RETURN inserted;
END

$BODY$
  LANGUAGE plpgsql VOLATILE
  COST 100;
ALTER FUNCTION "public"."populate_geometry_columns"("tbl_oid" oid, "use_typmod" bool) OWNER TO "postgres";

-- ----------------------------
-- Function structure for postgis_addbbox
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."postgis_addbbox"("public"."geometry");
CREATE FUNCTION "public"."postgis_addbbox"("public"."geometry")
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'LWGEOM_addBBOX'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."postgis_addbbox"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for postgis_cache_bbox
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."postgis_cache_bbox"();
CREATE FUNCTION "public"."postgis_cache_bbox"()
  RETURNS "pg_catalog"."trigger" AS '$libdir/postgis-3', 'cache_bbox'
  LANGUAGE c VOLATILE
  COST 1;
ALTER FUNCTION "public"."postgis_cache_bbox"() OWNER TO "postgres";

-- ----------------------------
-- Function structure for postgis_constraint_dims
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."postgis_constraint_dims"("geomschema" text, "geomtable" text, "geomcolumn" text);
CREATE FUNCTION "public"."postgis_constraint_dims"("geomschema" text, "geomtable" text, "geomcolumn" text)
  RETURNS "pg_catalog"."int4" AS $BODY$
SELECT  replace(split_part(s.consrc, ' = ', 2), ')', '')::integer
		 FROM pg_class c, pg_namespace n, pg_attribute a
		 , (SELECT connamespace, conrelid, conkey, pg_get_constraintdef(oid) As consrc
			FROM pg_constraint) AS s
		 WHERE n.nspname = $1
		 AND c.relname = $2
		 AND a.attname = $3
		 AND a.attrelid = c.oid
		 AND s.connamespace = n.oid
		 AND s.conrelid = c.oid
		 AND a.attnum = ANY (s.conkey)
		 AND s.consrc LIKE '%ndims(% = %';
$BODY$
  LANGUAGE sql STABLE STRICT
  COST 250;
ALTER FUNCTION "public"."postgis_constraint_dims"("geomschema" text, "geomtable" text, "geomcolumn" text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for postgis_constraint_srid
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."postgis_constraint_srid"("geomschema" text, "geomtable" text, "geomcolumn" text);
CREATE FUNCTION "public"."postgis_constraint_srid"("geomschema" text, "geomtable" text, "geomcolumn" text)
  RETURNS "pg_catalog"."int4" AS $BODY$
SELECT replace(replace(split_part(s.consrc, ' = ', 2), ')', ''), '(', '')::integer
		 FROM pg_class c, pg_namespace n, pg_attribute a
		 , (SELECT connamespace, conrelid, conkey, pg_get_constraintdef(oid) As consrc
			FROM pg_constraint) AS s
		 WHERE n.nspname = $1
		 AND c.relname = $2
		 AND a.attname = $3
		 AND a.attrelid = c.oid
		 AND s.connamespace = n.oid
		 AND s.conrelid = c.oid
		 AND a.attnum = ANY (s.conkey)
		 AND s.consrc LIKE '%srid(% = %';
$BODY$
  LANGUAGE sql STABLE STRICT
  COST 250;
ALTER FUNCTION "public"."postgis_constraint_srid"("geomschema" text, "geomtable" text, "geomcolumn" text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for postgis_constraint_type
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."postgis_constraint_type"("geomschema" text, "geomtable" text, "geomcolumn" text);
CREATE FUNCTION "public"."postgis_constraint_type"("geomschema" text, "geomtable" text, "geomcolumn" text)
  RETURNS "pg_catalog"."varchar" AS $BODY$
SELECT  replace(split_part(s.consrc, '''', 2), ')', '')::varchar
		 FROM pg_class c, pg_namespace n, pg_attribute a
		 , (SELECT connamespace, conrelid, conkey, pg_get_constraintdef(oid) As consrc
			FROM pg_constraint) AS s
		 WHERE n.nspname = $1
		 AND c.relname = $2
		 AND a.attname = $3
		 AND a.attrelid = c.oid
		 AND s.connamespace = n.oid
		 AND s.conrelid = c.oid
		 AND a.attnum = ANY (s.conkey)
		 AND s.consrc LIKE '%geometrytype(% = %';
$BODY$
  LANGUAGE sql STABLE STRICT
  COST 250;
ALTER FUNCTION "public"."postgis_constraint_type"("geomschema" text, "geomtable" text, "geomcolumn" text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for postgis_dropbbox
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."postgis_dropbbox"("public"."geometry");
CREATE FUNCTION "public"."postgis_dropbbox"("public"."geometry")
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'LWGEOM_dropBBOX'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."postgis_dropbbox"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for postgis_extensions_upgrade
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."postgis_extensions_upgrade"("target_version" text);
CREATE FUNCTION "public"."postgis_extensions_upgrade"("target_version" text=NULL::text)
  RETURNS "pg_catalog"."text" AS $BODY$
DECLARE
	rec record;
	sql text;
	var_schema text;
BEGIN

	FOR rec IN
		SELECT name, default_version, installed_version
		FROM pg_catalog.pg_available_extensions
		WHERE name IN (
			'postgis',
			'postgis_raster',
			'postgis_sfcgal',
			'postgis_topology',
			'postgis_tiger_geocoder'
		)
		ORDER BY length(name) -- this is to make sure 'postgis' is first !
	LOOP --{

		IF target_version IS NULL THEN
			target_version := rec.default_version;
		END IF;

		IF rec.installed_version IS NULL THEN --{
			-- If the support installed by available extension
			-- is found unpackaged, we package it
			IF --{
				 -- PostGIS is always available (this function is part of it)
				 rec.name = 'postgis'

				 -- PostGIS raster is available if type 'raster' exists
				 OR ( rec.name = 'postgis_raster' AND EXISTS (
							SELECT 1 FROM pg_catalog.pg_type
							WHERE typname = 'raster' ) )

				 -- PostGIS SFCGAL is available if
				 -- 'postgis_sfcgal_version' function exists
				 OR ( rec.name = 'postgis_sfcgal' AND EXISTS (
							SELECT 1 FROM pg_catalog.pg_proc
							WHERE proname = 'postgis_sfcgal_version' ) )

				 -- PostGIS Topology is available if
				 -- 'topology.topology' table exists
				 -- NOTE: watch out for https://trac.osgeo.org/postgis/ticket/2503
				 OR ( rec.name = 'postgis_topology' AND EXISTS (
							SELECT 1 FROM pg_catalog.pg_class c
							JOIN pg_catalog.pg_namespace n ON (c.relnamespace = n.oid )
							WHERE n.nspname = 'topology' AND c.relname = 'topology') )

				 OR ( rec.name = 'postgis_tiger_geocoder' AND EXISTS (
							SELECT 1 FROM pg_catalog.pg_class c
							JOIN pg_catalog.pg_namespace n ON (c.relnamespace = n.oid )
							WHERE n.nspname = 'tiger' AND c.relname = 'geocode_settings') )
			THEN --}{ -- the code is unpackaged
				-- Force install in same schema as postgis
				SELECT INTO var_schema n.nspname
				  FROM pg_namespace n, pg_proc p
				  WHERE p.proname = 'postgis_full_version'
					AND n.oid = p.pronamespace
				  LIMIT 1;
				IF rec.name NOT IN('postgis_topology', 'postgis_tiger_geocoder')
				THEN
					sql := format(
							  'CREATE EXTENSION %1$I SCHEMA %2$I VERSION unpackaged;'
							  'ALTER EXTENSION %1$I UPDATE TO %3$I',
							  rec.name, var_schema, target_version);
				ELSE
					sql := format(
							 'CREATE EXTENSION %1$I VERSION unpackaged;'
							 'ALTER EXTENSION %1$I UPDATE TO %2$I',
							 rec.name, target_version);
				END IF;
				RAISE NOTICE 'Packaging and updating %', rec.name;
				RAISE DEBUG '%', sql;
				EXECUTE sql;
			ELSE
				RAISE DEBUG 'Skipping % (not in use)', rec.name;
			END IF; --}
		ELSE -- The code is already packaged, upgrade it --}{
			sql = format(
				'ALTER EXTENSION %1$I UPDATE TO "ANY";'
				'ALTER EXTENSION %1$I UPDATE TO %2$I',
				rec.name, target_version
				);
			RAISE NOTICE 'Updating extension % %', rec.name, rec.installed_version;
			RAISE DEBUG '%', sql;
			EXECUTE sql;
		END IF; --}

	END LOOP; --}

	RETURN format(
		'Upgrade to version %s completed, run SELECT postgis_full_version(); for details',
		target_version
	);


END
$BODY$
  LANGUAGE plpgsql VOLATILE
  COST 100;
ALTER FUNCTION "public"."postgis_extensions_upgrade"("target_version" text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for postgis_full_version
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."postgis_full_version"();
CREATE FUNCTION "public"."postgis_full_version"()
  RETURNS "pg_catalog"."text" AS $BODY$
DECLARE
	libver text;
	librev text;
	projver text;
	projver_compiled text;
	geosver text;
	geosver_compiled text;
	sfcgalver text;
	gdalver text := NULL;
	libxmlver text;
	liblwgeomver text;
	dbproc text;
	relproc text;
	fullver text;
	rast_lib_ver text := NULL;
	rast_scr_ver text := NULL;
	topo_scr_ver text := NULL;
	json_lib_ver text;
	protobuf_lib_ver text;
	wagyu_lib_ver text;
	sfcgal_lib_ver text;
	sfcgal_scr_ver text;
	pgsql_scr_ver text;
	pgsql_ver text;
	core_is_extension bool;
BEGIN
	SELECT public.postgis_lib_version() INTO libver;
	SELECT public.postgis_proj_version() INTO projver;
	SELECT public.postgis_geos_version() INTO geosver;
	SELECT public.postgis_geos_compiled_version() INTO geosver_compiled;
	SELECT public.postgis_proj_compiled_version() INTO projver_compiled;
	SELECT public.postgis_libjson_version() INTO json_lib_ver;
	SELECT public.postgis_libprotobuf_version() INTO protobuf_lib_ver;
	SELECT public.postgis_wagyu_version() INTO wagyu_lib_ver;
	SELECT public._postgis_scripts_pgsql_version() INTO pgsql_scr_ver;
	SELECT public._postgis_pgsql_version() INTO pgsql_ver;
	BEGIN
		SELECT public.postgis_gdal_version() INTO gdalver;
	EXCEPTION
		WHEN undefined_function THEN
			RAISE DEBUG 'Function postgis_gdal_version() not found.  Is raster support enabled and rtpostgis.sql installed?';
	END;
	BEGIN
		SELECT public.postgis_sfcgal_full_version() INTO sfcgalver;
		BEGIN
			SELECT public.postgis_sfcgal_scripts_installed() INTO sfcgal_scr_ver;
		EXCEPTION
			WHEN undefined_function THEN
				sfcgal_scr_ver := 'missing';
		END;
	EXCEPTION
		WHEN undefined_function THEN
			RAISE DEBUG 'Function postgis_sfcgal_scripts_installed() not found. Is sfcgal support enabled and sfcgal.sql installed?';
	END;
	SELECT public.postgis_liblwgeom_version() INTO liblwgeomver;
	SELECT public.postgis_libxml_version() INTO libxmlver;
	SELECT public.postgis_scripts_installed() INTO dbproc;
	SELECT public.postgis_scripts_released() INTO relproc;
	SELECT public.postgis_lib_revision() INTO librev;
	BEGIN
		SELECT topology.postgis_topology_scripts_installed() INTO topo_scr_ver;
	EXCEPTION
		WHEN undefined_function OR invalid_schema_name THEN
			RAISE DEBUG 'Function postgis_topology_scripts_installed() not found. Is topology support enabled and topology.sql installed?';
		WHEN insufficient_privilege THEN
			RAISE NOTICE 'Topology support cannot be inspected. Is current user granted USAGE on schema "topology" ?';
		WHEN OTHERS THEN
			RAISE NOTICE 'Function postgis_topology_scripts_installed() could not be called: % (%)', SQLERRM, SQLSTATE;
	END;

	BEGIN
		SELECT postgis_raster_scripts_installed() INTO rast_scr_ver;
	EXCEPTION
		WHEN undefined_function THEN
			RAISE DEBUG 'Function postgis_raster_scripts_installed() not found. Is raster support enabled and rtpostgis.sql installed?';
		WHEN OTHERS THEN
			RAISE NOTICE 'Function postgis_raster_scripts_installed() could not be called: % (%)', SQLERRM, SQLSTATE;
	END;

	BEGIN
		SELECT public.postgis_raster_lib_version() INTO rast_lib_ver;
	EXCEPTION
		WHEN undefined_function THEN
			RAISE DEBUG 'Function postgis_raster_lib_version() not found. Is raster support enabled and rtpostgis.sql installed?';
		WHEN OTHERS THEN
			RAISE NOTICE 'Function postgis_raster_lib_version() could not be called: % (%)', SQLERRM, SQLSTATE;
	END;

	fullver = 'POSTGIS="' || libver;

	IF  librev IS NOT NULL THEN
		fullver = fullver || ' ' || librev;
	END IF;

	fullver = fullver || '"';

	IF EXISTS (
		SELECT * FROM pg_catalog.pg_extension
		WHERE extname = 'postgis')
	THEN
			fullver = fullver || ' [EXTENSION]';
			core_is_extension := true;
	ELSE
			core_is_extension := false;
	END IF;

	IF liblwgeomver != relproc THEN
		fullver = fullver || ' (liblwgeom version mismatch: "' || liblwgeomver || '")';
	END IF;

	fullver = fullver || ' PGSQL="' || pgsql_scr_ver || '"';
	IF pgsql_scr_ver != pgsql_ver THEN
		fullver = fullver || ' (procs need upgrade for use with PostgreSQL "' || pgsql_ver || '")';
	END IF;

	IF  geosver IS NOT NULL THEN
		fullver = fullver || ' GEOS="' || geosver || '"';
		IF (string_to_array(geosver, '.'))[1:2] != (string_to_array(geosver_compiled, '.'))[1:2]
		THEN
			fullver = format('%s (compiled against GEOS %s)', fullver, geosver_compiled);
		END IF;
	END IF;

	IF  sfcgalver IS NOT NULL THEN
		fullver = fullver || ' SFCGAL="' || sfcgalver || '"';
	END IF;

	IF  projver IS NOT NULL THEN
		fullver = fullver || ' PROJ="' || projver || '"';
		IF (string_to_array(projver, '.'))[1:3] != (string_to_array(projver_compiled, '.'))[1:3]
		THEN
			fullver = format('%s (compiled against PROJ %s)', fullver, projver_compiled);
		END IF;
	END IF;

	IF  gdalver IS NOT NULL THEN
		fullver = fullver || ' GDAL="' || gdalver || '"';
	END IF;

	IF  libxmlver IS NOT NULL THEN
		fullver = fullver || ' LIBXML="' || libxmlver || '"';
	END IF;

	IF json_lib_ver IS NOT NULL THEN
		fullver = fullver || ' LIBJSON="' || json_lib_ver || '"';
	END IF;

	IF protobuf_lib_ver IS NOT NULL THEN
		fullver = fullver || ' LIBPROTOBUF="' || protobuf_lib_ver || '"';
	END IF;

	IF wagyu_lib_ver IS NOT NULL THEN
		fullver = fullver || ' WAGYU="' || wagyu_lib_ver || '"';
	END IF;

	IF dbproc != relproc THEN
		fullver = fullver || ' (core procs from "' || dbproc || '" need upgrade)';
	END IF;

	IF topo_scr_ver IS NOT NULL THEN
		fullver = fullver || ' TOPOLOGY';
		IF topo_scr_ver != relproc THEN
			fullver = fullver || ' (topology procs from "' || topo_scr_ver || '" need upgrade)';
		END IF;
		IF core_is_extension AND NOT EXISTS (
			SELECT * FROM pg_catalog.pg_extension
			WHERE extname = 'postgis_topology')
		THEN
				fullver = fullver || ' [UNPACKAGED!]';
		END IF;
	END IF;

	IF rast_lib_ver IS NOT NULL THEN
		fullver = fullver || ' RASTER';
		IF rast_lib_ver != relproc THEN
			fullver = fullver || ' (raster lib from "' || rast_lib_ver || '" need upgrade)';
		END IF;
		IF core_is_extension AND NOT EXISTS (
			SELECT * FROM pg_catalog.pg_extension
			WHERE extname = 'postgis_raster')
		THEN
				fullver = fullver || ' [UNPACKAGED!]';
		END IF;
	END IF;

	IF rast_scr_ver IS NOT NULL AND rast_scr_ver != relproc THEN
		fullver = fullver || ' (raster procs from "' || rast_scr_ver || '" need upgrade)';
	END IF;

	IF sfcgal_scr_ver IS NOT NULL AND sfcgal_scr_ver != relproc THEN
		fullver = fullver || ' (sfcgal procs from "' || sfcgal_scr_ver || '" need upgrade)';
	END IF;

	-- Check for the presence of deprecated functions
	IF EXISTS ( SELECT oid FROM pg_catalog.pg_proc WHERE proname LIKE '%_deprecated_by_postgis_%' )
	THEN
		fullver = fullver || ' (deprecated functions exist, upgrade is not complete)';
	END IF;

	RETURN fullver;
END
$BODY$
  LANGUAGE plpgsql IMMUTABLE
  COST 100;
ALTER FUNCTION "public"."postgis_full_version"() OWNER TO "postgres";

-- ----------------------------
-- Function structure for postgis_geos_compiled_version
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."postgis_geos_compiled_version"();
CREATE FUNCTION "public"."postgis_geos_compiled_version"()
  RETURNS "pg_catalog"."text" AS '$libdir/postgis-3', 'postgis_geos_compiled_version'
  LANGUAGE c IMMUTABLE
  COST 1;
ALTER FUNCTION "public"."postgis_geos_compiled_version"() OWNER TO "postgres";

-- ----------------------------
-- Function structure for postgis_geos_noop
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."postgis_geos_noop"("public"."geometry");
CREATE FUNCTION "public"."postgis_geos_noop"("public"."geometry")
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'GEOSnoop'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."postgis_geos_noop"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for postgis_geos_version
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."postgis_geos_version"();
CREATE FUNCTION "public"."postgis_geos_version"()
  RETURNS "pg_catalog"."text" AS '$libdir/postgis-3', 'postgis_geos_version'
  LANGUAGE c IMMUTABLE
  COST 1;
ALTER FUNCTION "public"."postgis_geos_version"() OWNER TO "postgres";

-- ----------------------------
-- Function structure for postgis_getbbox
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."postgis_getbbox"("public"."geometry");
CREATE FUNCTION "public"."postgis_getbbox"("public"."geometry")
  RETURNS "public"."box2d" AS '$libdir/postgis-3', 'LWGEOM_to_BOX2DF'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."postgis_getbbox"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for postgis_hasbbox
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."postgis_hasbbox"("public"."geometry");
CREATE FUNCTION "public"."postgis_hasbbox"("public"."geometry")
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'LWGEOM_hasBBOX'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."postgis_hasbbox"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for postgis_index_supportfn
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."postgis_index_supportfn"(internal);
CREATE FUNCTION "public"."postgis_index_supportfn"(internal)
  RETURNS "pg_catalog"."internal" AS '$libdir/postgis-3', 'postgis_index_supportfn'
  LANGUAGE c VOLATILE
  COST 1;
ALTER FUNCTION "public"."postgis_index_supportfn"(internal) OWNER TO "postgres";

-- ----------------------------
-- Function structure for postgis_lib_build_date
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."postgis_lib_build_date"();
CREATE FUNCTION "public"."postgis_lib_build_date"()
  RETURNS "pg_catalog"."text" AS '$libdir/postgis-3', 'postgis_lib_build_date'
  LANGUAGE c IMMUTABLE
  COST 1;
ALTER FUNCTION "public"."postgis_lib_build_date"() OWNER TO "postgres";

-- ----------------------------
-- Function structure for postgis_lib_revision
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."postgis_lib_revision"();
CREATE FUNCTION "public"."postgis_lib_revision"()
  RETURNS "pg_catalog"."text" AS '$libdir/postgis-3', 'postgis_lib_revision'
  LANGUAGE c IMMUTABLE
  COST 1;
ALTER FUNCTION "public"."postgis_lib_revision"() OWNER TO "postgres";

-- ----------------------------
-- Function structure for postgis_lib_version
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."postgis_lib_version"();
CREATE FUNCTION "public"."postgis_lib_version"()
  RETURNS "pg_catalog"."text" AS '$libdir/postgis-3', 'postgis_lib_version'
  LANGUAGE c IMMUTABLE
  COST 1;
ALTER FUNCTION "public"."postgis_lib_version"() OWNER TO "postgres";

-- ----------------------------
-- Function structure for postgis_libjson_version
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."postgis_libjson_version"();
CREATE FUNCTION "public"."postgis_libjson_version"()
  RETURNS "pg_catalog"."text" AS '$libdir/postgis-3', 'postgis_libjson_version'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."postgis_libjson_version"() OWNER TO "postgres";

-- ----------------------------
-- Function structure for postgis_liblwgeom_version
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."postgis_liblwgeom_version"();
CREATE FUNCTION "public"."postgis_liblwgeom_version"()
  RETURNS "pg_catalog"."text" AS '$libdir/postgis-3', 'postgis_liblwgeom_version'
  LANGUAGE c IMMUTABLE
  COST 1;
ALTER FUNCTION "public"."postgis_liblwgeom_version"() OWNER TO "postgres";

-- ----------------------------
-- Function structure for postgis_libprotobuf_version
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."postgis_libprotobuf_version"();
CREATE FUNCTION "public"."postgis_libprotobuf_version"()
  RETURNS "pg_catalog"."text" AS '$libdir/postgis-3', 'postgis_libprotobuf_version'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."postgis_libprotobuf_version"() OWNER TO "postgres";

-- ----------------------------
-- Function structure for postgis_libxml_version
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."postgis_libxml_version"();
CREATE FUNCTION "public"."postgis_libxml_version"()
  RETURNS "pg_catalog"."text" AS '$libdir/postgis-3', 'postgis_libxml_version'
  LANGUAGE c IMMUTABLE
  COST 1;
ALTER FUNCTION "public"."postgis_libxml_version"() OWNER TO "postgres";

-- ----------------------------
-- Function structure for postgis_noop
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."postgis_noop"("public"."geometry");
CREATE FUNCTION "public"."postgis_noop"("public"."geometry")
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'LWGEOM_noop'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."postgis_noop"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for postgis_proj_compiled_version
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."postgis_proj_compiled_version"();
CREATE FUNCTION "public"."postgis_proj_compiled_version"()
  RETURNS "pg_catalog"."text" AS '$libdir/postgis-3', 'postgis_proj_compiled_version'
  LANGUAGE c IMMUTABLE
  COST 1;
ALTER FUNCTION "public"."postgis_proj_compiled_version"() OWNER TO "postgres";

-- ----------------------------
-- Function structure for postgis_proj_version
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."postgis_proj_version"();
CREATE FUNCTION "public"."postgis_proj_version"()
  RETURNS "pg_catalog"."text" AS '$libdir/postgis-3', 'postgis_proj_version'
  LANGUAGE c IMMUTABLE
  COST 1;
ALTER FUNCTION "public"."postgis_proj_version"() OWNER TO "postgres";

-- ----------------------------
-- Function structure for postgis_scripts_build_date
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."postgis_scripts_build_date"();
CREATE FUNCTION "public"."postgis_scripts_build_date"()
  RETURNS "pg_catalog"."text" AS $BODY$SELECT '2025-09-02 07:35:26'::text AS version$BODY$
  LANGUAGE sql IMMUTABLE
  COST 100;
ALTER FUNCTION "public"."postgis_scripts_build_date"() OWNER TO "postgres";

-- ----------------------------
-- Function structure for postgis_scripts_installed
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."postgis_scripts_installed"();
CREATE FUNCTION "public"."postgis_scripts_installed"()
  RETURNS "pg_catalog"."text" AS $BODY$ SELECT trim('3.6.0'::text || $rev$ 4c1967d $rev$) AS version $BODY$
  LANGUAGE sql IMMUTABLE
  COST 100;
ALTER FUNCTION "public"."postgis_scripts_installed"() OWNER TO "postgres";

-- ----------------------------
-- Function structure for postgis_scripts_released
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."postgis_scripts_released"();
CREATE FUNCTION "public"."postgis_scripts_released"()
  RETURNS "pg_catalog"."text" AS '$libdir/postgis-3', 'postgis_scripts_released'
  LANGUAGE c IMMUTABLE
  COST 1;
ALTER FUNCTION "public"."postgis_scripts_released"() OWNER TO "postgres";

-- ----------------------------
-- Function structure for postgis_srs
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."postgis_srs"("auth_name" text, "auth_srid" text);
CREATE FUNCTION "public"."postgis_srs"("auth_name" text, "auth_srid" text)
  RETURNS TABLE("auth_name" text, "auth_srid" text, "srname" text, "srtext" text, "proj4text" text, "point_sw" "public"."geometry", "point_ne" "public"."geometry") AS '$libdir/postgis-3', 'postgis_srs_entry'
  LANGUAGE c IMMUTABLE STRICT
  COST 250
  ROWS 1000;
ALTER FUNCTION "public"."postgis_srs"("auth_name" text, "auth_srid" text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for postgis_srs_all
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."postgis_srs_all"();
CREATE FUNCTION "public"."postgis_srs_all"()
  RETURNS TABLE("auth_name" text, "auth_srid" text, "srname" text, "srtext" text, "proj4text" text, "point_sw" "public"."geometry", "point_ne" "public"."geometry") AS '$libdir/postgis-3', 'postgis_srs_entry_all'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000
  ROWS 1000;
ALTER FUNCTION "public"."postgis_srs_all"() OWNER TO "postgres";

-- ----------------------------
-- Function structure for postgis_srs_codes
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."postgis_srs_codes"("auth_name" text);
CREATE FUNCTION "public"."postgis_srs_codes"("auth_name" text)
  RETURNS SETOF "pg_catalog"."text" AS '$libdir/postgis-3', 'postgis_srs_codes'
  LANGUAGE c IMMUTABLE STRICT
  COST 250
  ROWS 1000;
ALTER FUNCTION "public"."postgis_srs_codes"("auth_name" text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for postgis_srs_search
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."postgis_srs_search"("bounds" "public"."geometry", "authname" text);
CREATE FUNCTION "public"."postgis_srs_search"("bounds" "public"."geometry", "authname" text='EPSG'::text)
  RETURNS TABLE("auth_name" text, "auth_srid" text, "srname" text, "srtext" text, "proj4text" text, "point_sw" "public"."geometry", "point_ne" "public"."geometry") AS '$libdir/postgis-3', 'postgis_srs_search'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000
  ROWS 1000;
ALTER FUNCTION "public"."postgis_srs_search"("bounds" "public"."geometry", "authname" text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for postgis_svn_version
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."postgis_svn_version"();
CREATE FUNCTION "public"."postgis_svn_version"()
  RETURNS "pg_catalog"."text" AS $BODY$
	SELECT public._postgis_deprecate(
		'postgis_svn_version', 'postgis_lib_revision', '3.1.0');
	SELECT public.postgis_lib_revision();
$BODY$
  LANGUAGE sql IMMUTABLE
  COST 100;
ALTER FUNCTION "public"."postgis_svn_version"() OWNER TO "postgres";

-- ----------------------------
-- Function structure for postgis_transform_geometry
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."postgis_transform_geometry"("geom" "public"."geometry", text, text, int4);
CREATE FUNCTION "public"."postgis_transform_geometry"("geom" "public"."geometry", text, text, int4)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'transform_geom'
  LANGUAGE c IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."postgis_transform_geometry"("geom" "public"."geometry", text, text, int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for postgis_transform_pipeline_geometry
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."postgis_transform_pipeline_geometry"("geom" "public"."geometry", "pipeline" text, "forward" bool, "to_srid" int4);
CREATE FUNCTION "public"."postgis_transform_pipeline_geometry"("geom" "public"."geometry", "pipeline" text, "forward" bool, "to_srid" int4)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'transform_pipeline_geom'
  LANGUAGE c IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."postgis_transform_pipeline_geometry"("geom" "public"."geometry", "pipeline" text, "forward" bool, "to_srid" int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for postgis_type_name
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."postgis_type_name"("geomname" varchar, "coord_dimension" int4, "use_new_name" bool);
CREATE FUNCTION "public"."postgis_type_name"("geomname" varchar, "coord_dimension" int4, "use_new_name" bool=true)
  RETURNS "pg_catalog"."varchar" AS $BODY$
	SELECT CASE WHEN $3 THEN new_name ELSE old_name END As geomname
	FROM
	( VALUES
			('GEOMETRY', 'Geometry', 2),
			('GEOMETRY', 'GeometryZ', 3),
			('GEOMETRYM', 'GeometryM', 3),
			('GEOMETRY', 'GeometryZM', 4),

			('GEOMETRYCOLLECTION', 'GeometryCollection', 2),
			('GEOMETRYCOLLECTION', 'GeometryCollectionZ', 3),
			('GEOMETRYCOLLECTIONM', 'GeometryCollectionM', 3),
			('GEOMETRYCOLLECTION', 'GeometryCollectionZM', 4),

			('POINT', 'Point', 2),
			('POINT', 'PointZ', 3),
			('POINTM','PointM', 3),
			('POINT', 'PointZM', 4),

			('MULTIPOINT','MultiPoint', 2),
			('MULTIPOINT','MultiPointZ', 3),
			('MULTIPOINTM','MultiPointM', 3),
			('MULTIPOINT','MultiPointZM', 4),

			('POLYGON', 'Polygon', 2),
			('POLYGON', 'PolygonZ', 3),
			('POLYGONM', 'PolygonM', 3),
			('POLYGON', 'PolygonZM', 4),

			('MULTIPOLYGON', 'MultiPolygon', 2),
			('MULTIPOLYGON', 'MultiPolygonZ', 3),
			('MULTIPOLYGONM', 'MultiPolygonM', 3),
			('MULTIPOLYGON', 'MultiPolygonZM', 4),

			('MULTILINESTRING', 'MultiLineString', 2),
			('MULTILINESTRING', 'MultiLineStringZ', 3),
			('MULTILINESTRINGM', 'MultiLineStringM', 3),
			('MULTILINESTRING', 'MultiLineStringZM', 4),

			('LINESTRING', 'LineString', 2),
			('LINESTRING', 'LineStringZ', 3),
			('LINESTRINGM', 'LineStringM', 3),
			('LINESTRING', 'LineStringZM', 4),

			('CIRCULARSTRING', 'CircularString', 2),
			('CIRCULARSTRING', 'CircularStringZ', 3),
			('CIRCULARSTRINGM', 'CircularStringM' ,3),
			('CIRCULARSTRING', 'CircularStringZM', 4),

			('COMPOUNDCURVE', 'CompoundCurve', 2),
			('COMPOUNDCURVE', 'CompoundCurveZ', 3),
			('COMPOUNDCURVEM', 'CompoundCurveM', 3),
			('COMPOUNDCURVE', 'CompoundCurveZM', 4),

			('CURVEPOLYGON', 'CurvePolygon', 2),
			('CURVEPOLYGON', 'CurvePolygonZ', 3),
			('CURVEPOLYGONM', 'CurvePolygonM', 3),
			('CURVEPOLYGON', 'CurvePolygonZM', 4),

			('MULTICURVE', 'MultiCurve', 2),
			('MULTICURVE', 'MultiCurveZ', 3),
			('MULTICURVEM', 'MultiCurveM', 3),
			('MULTICURVE', 'MultiCurveZM', 4),

			('MULTISURFACE', 'MultiSurface', 2),
			('MULTISURFACE', 'MultiSurfaceZ', 3),
			('MULTISURFACEM', 'MultiSurfaceM', 3),
			('MULTISURFACE', 'MultiSurfaceZM', 4),

			('POLYHEDRALSURFACE', 'PolyhedralSurface', 2),
			('POLYHEDRALSURFACE', 'PolyhedralSurfaceZ', 3),
			('POLYHEDRALSURFACEM', 'PolyhedralSurfaceM', 3),
			('POLYHEDRALSURFACE', 'PolyhedralSurfaceZM', 4),

			('TRIANGLE', 'Triangle', 2),
			('TRIANGLE', 'TriangleZ', 3),
			('TRIANGLEM', 'TriangleM', 3),
			('TRIANGLE', 'TriangleZM', 4),

			('TIN', 'Tin', 2),
			('TIN', 'TinZ', 3),
			('TINM', 'TinM', 3),
			('TIN', 'TinZM', 4) )
			 As g(old_name, new_name, coord_dimension)
	WHERE (upper(old_name) = upper($1) OR upper(new_name) = upper($1))
		AND coord_dimension = $2;
$BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."postgis_type_name"("geomname" varchar, "coord_dimension" int4, "use_new_name" bool) OWNER TO "postgres";

-- ----------------------------
-- Function structure for postgis_typmod_dims
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."postgis_typmod_dims"(int4);
CREATE FUNCTION "public"."postgis_typmod_dims"(int4)
  RETURNS "pg_catalog"."int4" AS '$libdir/postgis-3', 'postgis_typmod_dims'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."postgis_typmod_dims"(int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for postgis_typmod_srid
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."postgis_typmod_srid"(int4);
CREATE FUNCTION "public"."postgis_typmod_srid"(int4)
  RETURNS "pg_catalog"."int4" AS '$libdir/postgis-3', 'postgis_typmod_srid'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."postgis_typmod_srid"(int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for postgis_typmod_type
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."postgis_typmod_type"(int4);
CREATE FUNCTION "public"."postgis_typmod_type"(int4)
  RETURNS "pg_catalog"."text" AS '$libdir/postgis-3', 'postgis_typmod_type'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."postgis_typmod_type"(int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for postgis_version
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."postgis_version"();
CREATE FUNCTION "public"."postgis_version"()
  RETURNS "pg_catalog"."text" AS '$libdir/postgis-3', 'postgis_version'
  LANGUAGE c IMMUTABLE
  COST 1;
ALTER FUNCTION "public"."postgis_version"() OWNER TO "postgres";

-- ----------------------------
-- Function structure for postgis_wagyu_version
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."postgis_wagyu_version"();
CREATE FUNCTION "public"."postgis_wagyu_version"()
  RETURNS "pg_catalog"."text" AS '$libdir/postgis-3', 'postgis_wagyu_version'
  LANGUAGE c IMMUTABLE
  COST 1;
ALTER FUNCTION "public"."postgis_wagyu_version"() OWNER TO "postgres";

-- ----------------------------
-- Function structure for set_limit
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."set_limit"(float4);
CREATE FUNCTION "public"."set_limit"(float4)
  RETURNS "pg_catalog"."float4" AS '$libdir/pg_trgm', 'set_limit'
  LANGUAGE c VOLATILE STRICT
  COST 1;
ALTER FUNCTION "public"."set_limit"(float4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for show_limit
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."show_limit"();
CREATE FUNCTION "public"."show_limit"()
  RETURNS "pg_catalog"."float4" AS '$libdir/pg_trgm', 'show_limit'
  LANGUAGE c STABLE STRICT
  COST 1;
ALTER FUNCTION "public"."show_limit"() OWNER TO "postgres";

-- ----------------------------
-- Function structure for show_trgm
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."show_trgm"(text);
CREATE FUNCTION "public"."show_trgm"(text)
  RETURNS "pg_catalog"."_text" AS '$libdir/pg_trgm', 'show_trgm'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."show_trgm"(text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for similarity
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."similarity"(text, text);
CREATE FUNCTION "public"."similarity"(text, text)
  RETURNS "pg_catalog"."float4" AS '$libdir/pg_trgm', 'similarity'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."similarity"(text, text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for similarity_dist
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."similarity_dist"(text, text);
CREATE FUNCTION "public"."similarity_dist"(text, text)
  RETURNS "pg_catalog"."float4" AS '$libdir/pg_trgm', 'similarity_dist'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."similarity_dist"(text, text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for similarity_op
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."similarity_op"(text, text);
CREATE FUNCTION "public"."similarity_op"(text, text)
  RETURNS "pg_catalog"."bool" AS '$libdir/pg_trgm', 'similarity_op'
  LANGUAGE c STABLE STRICT
  COST 1;
ALTER FUNCTION "public"."similarity_op"(text, text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for soundex
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."soundex"(text);
CREATE FUNCTION "public"."soundex"(text)
  RETURNS "pg_catalog"."text" AS '$libdir/fuzzystrmatch', 'soundex'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."soundex"(text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for sparsevec
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."sparsevec"("public"."sparsevec", int4, bool);
CREATE FUNCTION "public"."sparsevec"("public"."sparsevec", int4, bool)
  RETURNS "public"."sparsevec" AS '$libdir/vector', 'sparsevec'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."sparsevec"("public"."sparsevec", int4, bool) OWNER TO "postgres";

-- ----------------------------
-- Function structure for sparsevec_cmp
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."sparsevec_cmp"("public"."sparsevec", "public"."sparsevec");
CREATE FUNCTION "public"."sparsevec_cmp"("public"."sparsevec", "public"."sparsevec")
  RETURNS "pg_catalog"."int4" AS '$libdir/vector', 'sparsevec_cmp'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."sparsevec_cmp"("public"."sparsevec", "public"."sparsevec") OWNER TO "postgres";

-- ----------------------------
-- Function structure for sparsevec_eq
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."sparsevec_eq"("public"."sparsevec", "public"."sparsevec");
CREATE FUNCTION "public"."sparsevec_eq"("public"."sparsevec", "public"."sparsevec")
  RETURNS "pg_catalog"."bool" AS '$libdir/vector', 'sparsevec_eq'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."sparsevec_eq"("public"."sparsevec", "public"."sparsevec") OWNER TO "postgres";

-- ----------------------------
-- Function structure for sparsevec_ge
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."sparsevec_ge"("public"."sparsevec", "public"."sparsevec");
CREATE FUNCTION "public"."sparsevec_ge"("public"."sparsevec", "public"."sparsevec")
  RETURNS "pg_catalog"."bool" AS '$libdir/vector', 'sparsevec_ge'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."sparsevec_ge"("public"."sparsevec", "public"."sparsevec") OWNER TO "postgres";

-- ----------------------------
-- Function structure for sparsevec_gt
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."sparsevec_gt"("public"."sparsevec", "public"."sparsevec");
CREATE FUNCTION "public"."sparsevec_gt"("public"."sparsevec", "public"."sparsevec")
  RETURNS "pg_catalog"."bool" AS '$libdir/vector', 'sparsevec_gt'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."sparsevec_gt"("public"."sparsevec", "public"."sparsevec") OWNER TO "postgres";

-- ----------------------------
-- Function structure for sparsevec_in
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."sparsevec_in"(cstring, oid, int4);
CREATE FUNCTION "public"."sparsevec_in"(cstring, oid, int4)
  RETURNS "public"."sparsevec" AS '$libdir/vector', 'sparsevec_in'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."sparsevec_in"(cstring, oid, int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for sparsevec_l2_squared_distance
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."sparsevec_l2_squared_distance"("public"."sparsevec", "public"."sparsevec");
CREATE FUNCTION "public"."sparsevec_l2_squared_distance"("public"."sparsevec", "public"."sparsevec")
  RETURNS "pg_catalog"."float8" AS '$libdir/vector', 'sparsevec_l2_squared_distance'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."sparsevec_l2_squared_distance"("public"."sparsevec", "public"."sparsevec") OWNER TO "postgres";

-- ----------------------------
-- Function structure for sparsevec_le
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."sparsevec_le"("public"."sparsevec", "public"."sparsevec");
CREATE FUNCTION "public"."sparsevec_le"("public"."sparsevec", "public"."sparsevec")
  RETURNS "pg_catalog"."bool" AS '$libdir/vector', 'sparsevec_le'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."sparsevec_le"("public"."sparsevec", "public"."sparsevec") OWNER TO "postgres";

-- ----------------------------
-- Function structure for sparsevec_lt
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."sparsevec_lt"("public"."sparsevec", "public"."sparsevec");
CREATE FUNCTION "public"."sparsevec_lt"("public"."sparsevec", "public"."sparsevec")
  RETURNS "pg_catalog"."bool" AS '$libdir/vector', 'sparsevec_lt'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."sparsevec_lt"("public"."sparsevec", "public"."sparsevec") OWNER TO "postgres";

-- ----------------------------
-- Function structure for sparsevec_ne
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."sparsevec_ne"("public"."sparsevec", "public"."sparsevec");
CREATE FUNCTION "public"."sparsevec_ne"("public"."sparsevec", "public"."sparsevec")
  RETURNS "pg_catalog"."bool" AS '$libdir/vector', 'sparsevec_ne'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."sparsevec_ne"("public"."sparsevec", "public"."sparsevec") OWNER TO "postgres";

-- ----------------------------
-- Function structure for sparsevec_negative_inner_product
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."sparsevec_negative_inner_product"("public"."sparsevec", "public"."sparsevec");
CREATE FUNCTION "public"."sparsevec_negative_inner_product"("public"."sparsevec", "public"."sparsevec")
  RETURNS "pg_catalog"."float8" AS '$libdir/vector', 'sparsevec_negative_inner_product'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."sparsevec_negative_inner_product"("public"."sparsevec", "public"."sparsevec") OWNER TO "postgres";

-- ----------------------------
-- Function structure for sparsevec_out
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."sparsevec_out"("public"."sparsevec");
CREATE FUNCTION "public"."sparsevec_out"("public"."sparsevec")
  RETURNS "pg_catalog"."cstring" AS '$libdir/vector', 'sparsevec_out'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."sparsevec_out"("public"."sparsevec") OWNER TO "postgres";

-- ----------------------------
-- Function structure for sparsevec_recv
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."sparsevec_recv"(internal, oid, int4);
CREATE FUNCTION "public"."sparsevec_recv"(internal, oid, int4)
  RETURNS "public"."sparsevec" AS '$libdir/vector', 'sparsevec_recv'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."sparsevec_recv"(internal, oid, int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for sparsevec_send
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."sparsevec_send"("public"."sparsevec");
CREATE FUNCTION "public"."sparsevec_send"("public"."sparsevec")
  RETURNS "pg_catalog"."bytea" AS '$libdir/vector', 'sparsevec_send'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."sparsevec_send"("public"."sparsevec") OWNER TO "postgres";

-- ----------------------------
-- Function structure for sparsevec_to_halfvec
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."sparsevec_to_halfvec"("public"."sparsevec", int4, bool);
CREATE FUNCTION "public"."sparsevec_to_halfvec"("public"."sparsevec", int4, bool)
  RETURNS "public"."halfvec" AS '$libdir/vector', 'sparsevec_to_halfvec'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."sparsevec_to_halfvec"("public"."sparsevec", int4, bool) OWNER TO "postgres";

-- ----------------------------
-- Function structure for sparsevec_to_vector
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."sparsevec_to_vector"("public"."sparsevec", int4, bool);
CREATE FUNCTION "public"."sparsevec_to_vector"("public"."sparsevec", int4, bool)
  RETURNS "public"."vector" AS '$libdir/vector', 'sparsevec_to_vector'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."sparsevec_to_vector"("public"."sparsevec", int4, bool) OWNER TO "postgres";

-- ----------------------------
-- Function structure for sparsevec_typmod_in
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."sparsevec_typmod_in"(_cstring);
CREATE FUNCTION "public"."sparsevec_typmod_in"(_cstring)
  RETURNS "pg_catalog"."int4" AS '$libdir/vector', 'sparsevec_typmod_in'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."sparsevec_typmod_in"(_cstring) OWNER TO "postgres";

-- ----------------------------
-- Function structure for spheroid_in
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."spheroid_in"(cstring);
CREATE FUNCTION "public"."spheroid_in"(cstring)
  RETURNS "public"."spheroid" AS '$libdir/postgis-3', 'ellipsoid_in'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."spheroid_in"(cstring) OWNER TO "postgres";

-- ----------------------------
-- Function structure for spheroid_out
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."spheroid_out"("public"."spheroid");
CREATE FUNCTION "public"."spheroid_out"("public"."spheroid")
  RETURNS "pg_catalog"."cstring" AS '$libdir/postgis-3', 'ellipsoid_out'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."spheroid_out"("public"."spheroid") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_3dclosestpoint
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_3dclosestpoint"("geom1" "public"."geometry", "geom2" "public"."geometry");
CREATE FUNCTION "public"."st_3dclosestpoint"("geom1" "public"."geometry", "geom2" "public"."geometry")
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'LWGEOM_closestpoint3d'
  LANGUAGE c IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_3dclosestpoint"("geom1" "public"."geometry", "geom2" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_3ddfullywithin
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_3ddfullywithin"("geom1" "public"."geometry", "geom2" "public"."geometry", float8);
CREATE FUNCTION "public"."st_3ddfullywithin"("geom1" "public"."geometry", "geom2" "public"."geometry", float8)
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'LWGEOM_dfullywithin3d'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_3ddfullywithin"("geom1" "public"."geometry", "geom2" "public"."geometry", float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_3ddistance
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_3ddistance"("geom1" "public"."geometry", "geom2" "public"."geometry");
CREATE FUNCTION "public"."st_3ddistance"("geom1" "public"."geometry", "geom2" "public"."geometry")
  RETURNS "pg_catalog"."float8" AS '$libdir/postgis-3', 'ST_3DDistance'
  LANGUAGE c IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_3ddistance"("geom1" "public"."geometry", "geom2" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_3ddwithin
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_3ddwithin"("geom1" "public"."geometry", "geom2" "public"."geometry", float8);
CREATE FUNCTION "public"."st_3ddwithin"("geom1" "public"."geometry", "geom2" "public"."geometry", float8)
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'LWGEOM_dwithin3d'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_3ddwithin"("geom1" "public"."geometry", "geom2" "public"."geometry", float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_3dintersects
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_3dintersects"("geom1" "public"."geometry", "geom2" "public"."geometry");
CREATE FUNCTION "public"."st_3dintersects"("geom1" "public"."geometry", "geom2" "public"."geometry")
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'ST_3DIntersects'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_3dintersects"("geom1" "public"."geometry", "geom2" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_3dlength
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_3dlength"("public"."geometry");
CREATE FUNCTION "public"."st_3dlength"("public"."geometry")
  RETURNS "pg_catalog"."float8" AS '$libdir/postgis-3', 'LWGEOM_length_linestring'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_3dlength"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_3dlineinterpolatepoint
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_3dlineinterpolatepoint"("public"."geometry", float8);
CREATE FUNCTION "public"."st_3dlineinterpolatepoint"("public"."geometry", float8)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'ST_3DLineInterpolatePoint'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_3dlineinterpolatepoint"("public"."geometry", float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_3dlongestline
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_3dlongestline"("geom1" "public"."geometry", "geom2" "public"."geometry");
CREATE FUNCTION "public"."st_3dlongestline"("geom1" "public"."geometry", "geom2" "public"."geometry")
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'LWGEOM_longestline3d'
  LANGUAGE c IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_3dlongestline"("geom1" "public"."geometry", "geom2" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_3dmakebox
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_3dmakebox"("geom1" "public"."geometry", "geom2" "public"."geometry");
CREATE FUNCTION "public"."st_3dmakebox"("geom1" "public"."geometry", "geom2" "public"."geometry")
  RETURNS "public"."box3d" AS '$libdir/postgis-3', 'BOX3D_construct'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_3dmakebox"("geom1" "public"."geometry", "geom2" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_3dmaxdistance
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_3dmaxdistance"("geom1" "public"."geometry", "geom2" "public"."geometry");
CREATE FUNCTION "public"."st_3dmaxdistance"("geom1" "public"."geometry", "geom2" "public"."geometry")
  RETURNS "pg_catalog"."float8" AS '$libdir/postgis-3', 'LWGEOM_maxdistance3d'
  LANGUAGE c IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_3dmaxdistance"("geom1" "public"."geometry", "geom2" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_3dperimeter
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_3dperimeter"("public"."geometry");
CREATE FUNCTION "public"."st_3dperimeter"("public"."geometry")
  RETURNS "pg_catalog"."float8" AS '$libdir/postgis-3', 'LWGEOM_perimeter_poly'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_3dperimeter"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_3dshortestline
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_3dshortestline"("geom1" "public"."geometry", "geom2" "public"."geometry");
CREATE FUNCTION "public"."st_3dshortestline"("geom1" "public"."geometry", "geom2" "public"."geometry")
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'LWGEOM_shortestline3d'
  LANGUAGE c IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_3dshortestline"("geom1" "public"."geometry", "geom2" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_addmeasure
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_addmeasure"("public"."geometry", float8, float8);
CREATE FUNCTION "public"."st_addmeasure"("public"."geometry", float8, float8)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'ST_AddMeasure'
  LANGUAGE c IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_addmeasure"("public"."geometry", float8, float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_addpoint
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_addpoint"("geom1" "public"."geometry", "geom2" "public"."geometry", int4);
CREATE FUNCTION "public"."st_addpoint"("geom1" "public"."geometry", "geom2" "public"."geometry", int4)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'LWGEOM_addpoint'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_addpoint"("geom1" "public"."geometry", "geom2" "public"."geometry", int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_addpoint
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_addpoint"("geom1" "public"."geometry", "geom2" "public"."geometry");
CREATE FUNCTION "public"."st_addpoint"("geom1" "public"."geometry", "geom2" "public"."geometry")
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'LWGEOM_addpoint'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_addpoint"("geom1" "public"."geometry", "geom2" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_affine
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_affine"("public"."geometry", float8, float8, float8, float8, float8, float8, float8, float8, float8, float8, float8, float8);
CREATE FUNCTION "public"."st_affine"("public"."geometry", float8, float8, float8, float8, float8, float8, float8, float8, float8, float8, float8, float8)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'LWGEOM_affine'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_affine"("public"."geometry", float8, float8, float8, float8, float8, float8, float8, float8, float8, float8, float8, float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_affine
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_affine"("public"."geometry", float8, float8, float8, float8, float8, float8);
CREATE FUNCTION "public"."st_affine"("public"."geometry", float8, float8, float8, float8, float8, float8)
  RETURNS "public"."geometry" AS $BODY$SELECT public.ST_Affine($1,  $2, $3, 0,  $4, $5, 0,  0, 0, 1,  $6, $7, 0)$BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_affine"("public"."geometry", float8, float8, float8, float8, float8, float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_angle
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_angle"("pt1" "public"."geometry", "pt2" "public"."geometry", "pt3" "public"."geometry", "pt4" "public"."geometry");
CREATE FUNCTION "public"."st_angle"("pt1" "public"."geometry", "pt2" "public"."geometry", "pt3" "public"."geometry", "pt4" "public"."geometry"='0101000000000000000000F87F000000000000F87F'::geometry)
  RETURNS "pg_catalog"."float8" AS '$libdir/postgis-3', 'LWGEOM_angle'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_angle"("pt1" "public"."geometry", "pt2" "public"."geometry", "pt3" "public"."geometry", "pt4" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_angle
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_angle"("line1" "public"."geometry", "line2" "public"."geometry");
CREATE FUNCTION "public"."st_angle"("line1" "public"."geometry", "line2" "public"."geometry")
  RETURNS "pg_catalog"."float8" AS $BODY$SELECT public.ST_Angle(public.St_StartPoint($1), public.ST_EndPoint($1), public.ST_StartPoint($2), public.ST_EndPoint($2))$BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_angle"("line1" "public"."geometry", "line2" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_area
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_area"(text);
CREATE FUNCTION "public"."st_area"(text)
  RETURNS "pg_catalog"."float8" AS $BODY$ SELECT public.ST_Area($1::public.geometry);  $BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 100;
ALTER FUNCTION "public"."st_area"(text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_area
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_area"("geog" "public"."geography", "use_spheroid" bool);
CREATE FUNCTION "public"."st_area"("geog" "public"."geography", "use_spheroid" bool=true)
  RETURNS "pg_catalog"."float8" AS '$libdir/postgis-3', 'geography_area'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_area"("geog" "public"."geography", "use_spheroid" bool) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_area
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_area"("public"."geometry");
CREATE FUNCTION "public"."st_area"("public"."geometry")
  RETURNS "pg_catalog"."float8" AS '$libdir/postgis-3', 'ST_Area'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_area"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_area2d
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_area2d"("public"."geometry");
CREATE FUNCTION "public"."st_area2d"("public"."geometry")
  RETURNS "pg_catalog"."float8" AS '$libdir/postgis-3', 'ST_Area'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_area2d"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_asbinary
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_asbinary"("public"."geography", text);
CREATE FUNCTION "public"."st_asbinary"("public"."geography", text)
  RETURNS "pg_catalog"."bytea" AS '$libdir/postgis-3', 'LWGEOM_asBinary'
  LANGUAGE c IMMUTABLE
  COST 50;
ALTER FUNCTION "public"."st_asbinary"("public"."geography", text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_asbinary
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_asbinary"("public"."geometry");
CREATE FUNCTION "public"."st_asbinary"("public"."geometry")
  RETURNS "pg_catalog"."bytea" AS '$libdir/postgis-3', 'LWGEOM_asBinary'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_asbinary"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_asbinary
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_asbinary"("public"."geography");
CREATE FUNCTION "public"."st_asbinary"("public"."geography")
  RETURNS "pg_catalog"."bytea" AS '$libdir/postgis-3', 'LWGEOM_asBinary'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_asbinary"("public"."geography") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_asbinary
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_asbinary"("public"."geometry", text);
CREATE FUNCTION "public"."st_asbinary"("public"."geometry", text)
  RETURNS "pg_catalog"."bytea" AS '$libdir/postgis-3', 'LWGEOM_asBinary'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_asbinary"("public"."geometry", text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_asencodedpolyline
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_asencodedpolyline"("geom" "public"."geometry", "nprecision" int4);
CREATE FUNCTION "public"."st_asencodedpolyline"("geom" "public"."geometry", "nprecision" int4=5)
  RETURNS "pg_catalog"."text" AS '$libdir/postgis-3', 'LWGEOM_asEncodedPolyline'
  LANGUAGE c IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_asencodedpolyline"("geom" "public"."geometry", "nprecision" int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_asewkb
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_asewkb"("public"."geometry", text);
CREATE FUNCTION "public"."st_asewkb"("public"."geometry", text)
  RETURNS "pg_catalog"."bytea" AS '$libdir/postgis-3', 'WKBFromLWGEOM'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_asewkb"("public"."geometry", text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_asewkb
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_asewkb"("public"."geometry");
CREATE FUNCTION "public"."st_asewkb"("public"."geometry")
  RETURNS "pg_catalog"."bytea" AS '$libdir/postgis-3', 'WKBFromLWGEOM'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_asewkb"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_asewkt
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_asewkt"("public"."geometry");
CREATE FUNCTION "public"."st_asewkt"("public"."geometry")
  RETURNS "pg_catalog"."text" AS '$libdir/postgis-3', 'LWGEOM_asEWKT'
  LANGUAGE c IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_asewkt"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_asewkt
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_asewkt"("public"."geometry", int4);
CREATE FUNCTION "public"."st_asewkt"("public"."geometry", int4)
  RETURNS "pg_catalog"."text" AS '$libdir/postgis-3', 'LWGEOM_asEWKT'
  LANGUAGE c IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_asewkt"("public"."geometry", int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_asewkt
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_asewkt"(text);
CREATE FUNCTION "public"."st_asewkt"(text)
  RETURNS "pg_catalog"."text" AS $BODY$ SELECT public.ST_AsEWKT($1::public.geometry);  $BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_asewkt"(text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_asewkt
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_asewkt"("public"."geography", int4);
CREATE FUNCTION "public"."st_asewkt"("public"."geography", int4)
  RETURNS "pg_catalog"."text" AS '$libdir/postgis-3', 'LWGEOM_asEWKT'
  LANGUAGE c IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_asewkt"("public"."geography", int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_asewkt
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_asewkt"("public"."geography");
CREATE FUNCTION "public"."st_asewkt"("public"."geography")
  RETURNS "pg_catalog"."text" AS '$libdir/postgis-3', 'LWGEOM_asEWKT'
  LANGUAGE c IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_asewkt"("public"."geography") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_asgeojson
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_asgeojson"("r" record, "geom_column" text, "maxdecimaldigits" int4, "pretty_bool" bool, "id_column" text);
CREATE FUNCTION "public"."st_asgeojson"("r" record, "geom_column" text=''::text, "maxdecimaldigits" int4=9, "pretty_bool" bool=false, "id_column" text=''::text)
  RETURNS "pg_catalog"."text" AS '$libdir/postgis-3', 'ST_AsGeoJsonRow'
  LANGUAGE c STABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_asgeojson"("r" record, "geom_column" text, "maxdecimaldigits" int4, "pretty_bool" bool, "id_column" text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_asgeojson
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_asgeojson"(text);
CREATE FUNCTION "public"."st_asgeojson"(text)
  RETURNS "pg_catalog"."text" AS $BODY$ SELECT public.ST_AsGeoJson($1::public.geometry, 9, 0);  $BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_asgeojson"(text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_asgeojson
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_asgeojson"("geog" "public"."geography", "maxdecimaldigits" int4, "options" int4);
CREATE FUNCTION "public"."st_asgeojson"("geog" "public"."geography", "maxdecimaldigits" int4=9, "options" int4=0)
  RETURNS "pg_catalog"."text" AS '$libdir/postgis-3', 'geography_as_geojson'
  LANGUAGE c IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_asgeojson"("geog" "public"."geography", "maxdecimaldigits" int4, "options" int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_asgeojson
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_asgeojson"("geom" "public"."geometry", "maxdecimaldigits" int4, "options" int4);
CREATE FUNCTION "public"."st_asgeojson"("geom" "public"."geometry", "maxdecimaldigits" int4=9, "options" int4=8)
  RETURNS "pg_catalog"."text" AS '$libdir/postgis-3', 'LWGEOM_asGeoJson'
  LANGUAGE c IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_asgeojson"("geom" "public"."geometry", "maxdecimaldigits" int4, "options" int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_asgml
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_asgml"("version" int4, "geom" "public"."geometry", "maxdecimaldigits" int4, "options" int4, "nprefix" text, "id" text);
CREATE FUNCTION "public"."st_asgml"("version" int4, "geom" "public"."geometry", "maxdecimaldigits" int4=15, "options" int4=0, "nprefix" text=NULL::text, "id" text=NULL::text)
  RETURNS "pg_catalog"."text" AS '$libdir/postgis-3', 'LWGEOM_asGML'
  LANGUAGE c IMMUTABLE
  COST 250;
ALTER FUNCTION "public"."st_asgml"("version" int4, "geom" "public"."geometry", "maxdecimaldigits" int4, "options" int4, "nprefix" text, "id" text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_asgml
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_asgml"("version" int4, "geog" "public"."geography", "maxdecimaldigits" int4, "options" int4, "nprefix" text, "id" text);
CREATE FUNCTION "public"."st_asgml"("version" int4, "geog" "public"."geography", "maxdecimaldigits" int4=15, "options" int4=0, "nprefix" text='gml'::text, "id" text=''::text)
  RETURNS "pg_catalog"."text" AS '$libdir/postgis-3', 'geography_as_gml'
  LANGUAGE c IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_asgml"("version" int4, "geog" "public"."geography", "maxdecimaldigits" int4, "options" int4, "nprefix" text, "id" text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_asgml
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_asgml"("geog" "public"."geography", "maxdecimaldigits" int4, "options" int4, "nprefix" text, "id" text);
CREATE FUNCTION "public"."st_asgml"("geog" "public"."geography", "maxdecimaldigits" int4=15, "options" int4=0, "nprefix" text='gml'::text, "id" text=''::text)
  RETURNS "pg_catalog"."text" AS '$libdir/postgis-3', 'geography_as_gml'
  LANGUAGE c IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_asgml"("geog" "public"."geography", "maxdecimaldigits" int4, "options" int4, "nprefix" text, "id" text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_asgml
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_asgml"(text);
CREATE FUNCTION "public"."st_asgml"(text)
  RETURNS "pg_catalog"."text" AS $BODY$ SELECT public._ST_AsGML(2,$1::public.geometry,15,0, NULL, NULL);  $BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_asgml"(text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_asgml
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_asgml"("geom" "public"."geometry", "maxdecimaldigits" int4, "options" int4);
CREATE FUNCTION "public"."st_asgml"("geom" "public"."geometry", "maxdecimaldigits" int4=15, "options" int4=0)
  RETURNS "pg_catalog"."text" AS '$libdir/postgis-3', 'LWGEOM_asGML'
  LANGUAGE c IMMUTABLE
  COST 250;
ALTER FUNCTION "public"."st_asgml"("geom" "public"."geometry", "maxdecimaldigits" int4, "options" int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_ashexewkb
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_ashexewkb"("public"."geometry");
CREATE FUNCTION "public"."st_ashexewkb"("public"."geometry")
  RETURNS "pg_catalog"."text" AS '$libdir/postgis-3', 'LWGEOM_asHEXEWKB'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_ashexewkb"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_ashexewkb
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_ashexewkb"("public"."geometry", text);
CREATE FUNCTION "public"."st_ashexewkb"("public"."geometry", text)
  RETURNS "pg_catalog"."text" AS '$libdir/postgis-3', 'LWGEOM_asHEXEWKB'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_ashexewkb"("public"."geometry", text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_askml
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_askml"(text);
CREATE FUNCTION "public"."st_askml"(text)
  RETURNS "pg_catalog"."text" AS $BODY$ SELECT public.ST_AsKML($1::public.geometry, 15);  $BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_askml"(text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_askml
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_askml"("geom" "public"."geometry", "maxdecimaldigits" int4, "nprefix" text);
CREATE FUNCTION "public"."st_askml"("geom" "public"."geometry", "maxdecimaldigits" int4=15, "nprefix" text=''::text)
  RETURNS "pg_catalog"."text" AS '$libdir/postgis-3', 'LWGEOM_asKML'
  LANGUAGE c IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_askml"("geom" "public"."geometry", "maxdecimaldigits" int4, "nprefix" text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_askml
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_askml"("geog" "public"."geography", "maxdecimaldigits" int4, "nprefix" text);
CREATE FUNCTION "public"."st_askml"("geog" "public"."geography", "maxdecimaldigits" int4=15, "nprefix" text=''::text)
  RETURNS "pg_catalog"."text" AS '$libdir/postgis-3', 'geography_as_kml'
  LANGUAGE c IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_askml"("geog" "public"."geography", "maxdecimaldigits" int4, "nprefix" text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_aslatlontext
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_aslatlontext"("geom" "public"."geometry", "tmpl" text);
CREATE FUNCTION "public"."st_aslatlontext"("geom" "public"."geometry", "tmpl" text=''::text)
  RETURNS "pg_catalog"."text" AS '$libdir/postgis-3', 'LWGEOM_to_latlon'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_aslatlontext"("geom" "public"."geometry", "tmpl" text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_asmarc21
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_asmarc21"("geom" "public"."geometry", "format" text);
CREATE FUNCTION "public"."st_asmarc21"("geom" "public"."geometry", "format" text='hdddmmss'::text)
  RETURNS "pg_catalog"."text" AS '$libdir/postgis-3', 'ST_AsMARC21'
  LANGUAGE c IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_asmarc21"("geom" "public"."geometry", "format" text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_asmvtgeom
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_asmvtgeom"("geom" "public"."geometry", "bounds" "public"."box2d", "extent" int4, "buffer" int4, "clip_geom" bool);
CREATE FUNCTION "public"."st_asmvtgeom"("geom" "public"."geometry", "bounds" "public"."box2d", "extent" int4=4096, "buffer" int4=256, "clip_geom" bool=true)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'ST_AsMVTGeom'
  LANGUAGE c IMMUTABLE
  COST 250;
ALTER FUNCTION "public"."st_asmvtgeom"("geom" "public"."geometry", "bounds" "public"."box2d", "extent" int4, "buffer" int4, "clip_geom" bool) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_assvg
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_assvg"("geog" "public"."geography", "rel" int4, "maxdecimaldigits" int4);
CREATE FUNCTION "public"."st_assvg"("geog" "public"."geography", "rel" int4=0, "maxdecimaldigits" int4=15)
  RETURNS "pg_catalog"."text" AS '$libdir/postgis-3', 'geography_as_svg'
  LANGUAGE c IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_assvg"("geog" "public"."geography", "rel" int4, "maxdecimaldigits" int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_assvg
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_assvg"("geom" "public"."geometry", "rel" int4, "maxdecimaldigits" int4);
CREATE FUNCTION "public"."st_assvg"("geom" "public"."geometry", "rel" int4=0, "maxdecimaldigits" int4=15)
  RETURNS "pg_catalog"."text" AS '$libdir/postgis-3', 'LWGEOM_asSVG'
  LANGUAGE c IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_assvg"("geom" "public"."geometry", "rel" int4, "maxdecimaldigits" int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_assvg
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_assvg"(text);
CREATE FUNCTION "public"."st_assvg"(text)
  RETURNS "pg_catalog"."text" AS $BODY$ SELECT public.ST_AsSVG($1::public.geometry,0,15);  $BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_assvg"(text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_astext
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_astext"("public"."geography");
CREATE FUNCTION "public"."st_astext"("public"."geography")
  RETURNS "pg_catalog"."text" AS '$libdir/postgis-3', 'LWGEOM_asText'
  LANGUAGE c IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_astext"("public"."geography") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_astext
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_astext"(text);
CREATE FUNCTION "public"."st_astext"(text)
  RETURNS "pg_catalog"."text" AS $BODY$ SELECT public.ST_AsText($1::public.geometry);  $BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_astext"(text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_astext
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_astext"("public"."geography", int4);
CREATE FUNCTION "public"."st_astext"("public"."geography", int4)
  RETURNS "pg_catalog"."text" AS '$libdir/postgis-3', 'LWGEOM_asText'
  LANGUAGE c IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_astext"("public"."geography", int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_astext
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_astext"("public"."geometry", int4);
CREATE FUNCTION "public"."st_astext"("public"."geometry", int4)
  RETURNS "pg_catalog"."text" AS '$libdir/postgis-3', 'LWGEOM_asText'
  LANGUAGE c IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_astext"("public"."geometry", int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_astext
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_astext"("public"."geometry");
CREATE FUNCTION "public"."st_astext"("public"."geometry")
  RETURNS "pg_catalog"."text" AS '$libdir/postgis-3', 'LWGEOM_asText'
  LANGUAGE c IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_astext"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_astwkb
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_astwkb"("geom" "public"."_geometry", "ids" _int8, "prec" int4, "prec_z" int4, "prec_m" int4, "with_sizes" bool, "with_boxes" bool);
CREATE FUNCTION "public"."st_astwkb"("geom" "public"."_geometry", "ids" _int8, "prec" int4=NULL::integer, "prec_z" int4=NULL::integer, "prec_m" int4=NULL::integer, "with_sizes" bool=NULL::boolean, "with_boxes" bool=NULL::boolean)
  RETURNS "pg_catalog"."bytea" AS '$libdir/postgis-3', 'TWKBFromLWGEOMArray'
  LANGUAGE c IMMUTABLE
  COST 50;
ALTER FUNCTION "public"."st_astwkb"("geom" "public"."_geometry", "ids" _int8, "prec" int4, "prec_z" int4, "prec_m" int4, "with_sizes" bool, "with_boxes" bool) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_astwkb
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_astwkb"("geom" "public"."geometry", "prec" int4, "prec_z" int4, "prec_m" int4, "with_sizes" bool, "with_boxes" bool);
CREATE FUNCTION "public"."st_astwkb"("geom" "public"."geometry", "prec" int4=NULL::integer, "prec_z" int4=NULL::integer, "prec_m" int4=NULL::integer, "with_sizes" bool=NULL::boolean, "with_boxes" bool=NULL::boolean)
  RETURNS "pg_catalog"."bytea" AS '$libdir/postgis-3', 'TWKBFromLWGEOM'
  LANGUAGE c IMMUTABLE
  COST 50;
ALTER FUNCTION "public"."st_astwkb"("geom" "public"."geometry", "prec" int4, "prec_z" int4, "prec_m" int4, "with_sizes" bool, "with_boxes" bool) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_asx3d
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_asx3d"("geom" "public"."geometry", "maxdecimaldigits" int4, "options" int4);
CREATE FUNCTION "public"."st_asx3d"("geom" "public"."geometry", "maxdecimaldigits" int4=15, "options" int4=0)
  RETURNS "pg_catalog"."text" AS $BODY$SELECT public._ST_AsX3D(3,$1,$2,$3,'');$BODY$
  LANGUAGE sql IMMUTABLE
  COST 250;
ALTER FUNCTION "public"."st_asx3d"("geom" "public"."geometry", "maxdecimaldigits" int4, "options" int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_azimuth
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_azimuth"("geog1" "public"."geography", "geog2" "public"."geography");
CREATE FUNCTION "public"."st_azimuth"("geog1" "public"."geography", "geog2" "public"."geography")
  RETURNS "pg_catalog"."float8" AS '$libdir/postgis-3', 'geography_azimuth'
  LANGUAGE c IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_azimuth"("geog1" "public"."geography", "geog2" "public"."geography") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_azimuth
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_azimuth"("geom1" "public"."geometry", "geom2" "public"."geometry");
CREATE FUNCTION "public"."st_azimuth"("geom1" "public"."geometry", "geom2" "public"."geometry")
  RETURNS "pg_catalog"."float8" AS '$libdir/postgis-3', 'LWGEOM_azimuth'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_azimuth"("geom1" "public"."geometry", "geom2" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_bdmpolyfromtext
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_bdmpolyfromtext"(text, int4);
CREATE FUNCTION "public"."st_bdmpolyfromtext"(text, int4)
  RETURNS "public"."geometry" AS $BODY$
DECLARE
	geomtext alias for $1;
	srid alias for $2;
	mline public.geometry;
	geom public.geometry;
BEGIN
	mline := public.ST_MultiLineStringFromText(geomtext, srid);

	IF mline IS NULL
	THEN
		RAISE EXCEPTION 'Input is not a MultiLinestring';
	END IF;

	geom := public.ST_Multi(public.ST_BuildArea(mline));

	RETURN geom;
END;
$BODY$
  LANGUAGE plpgsql IMMUTABLE STRICT
  COST 100;
ALTER FUNCTION "public"."st_bdmpolyfromtext"(text, int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_bdpolyfromtext
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_bdpolyfromtext"(text, int4);
CREATE FUNCTION "public"."st_bdpolyfromtext"(text, int4)
  RETURNS "public"."geometry" AS $BODY$
DECLARE
	geomtext alias for $1;
	srid alias for $2;
	mline public.geometry;
	geom public.geometry;
BEGIN
	mline := public.ST_MultiLineStringFromText(geomtext, srid);

	IF mline IS NULL
	THEN
		RAISE EXCEPTION 'Input is not a MultiLinestring';
	END IF;

	geom := public.ST_BuildArea(mline);

	IF public.ST_GeometryType(geom) != 'ST_Polygon'
	THEN
		RAISE EXCEPTION 'Input returns more then a single polygon, try using BdMPolyFromText instead';
	END IF;

	RETURN geom;
END;
$BODY$
  LANGUAGE plpgsql IMMUTABLE STRICT
  COST 100;
ALTER FUNCTION "public"."st_bdpolyfromtext"(text, int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_boundary
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_boundary"("public"."geometry");
CREATE FUNCTION "public"."st_boundary"("public"."geometry")
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'boundary'
  LANGUAGE c IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_boundary"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_boundingdiagonal
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_boundingdiagonal"("geom" "public"."geometry", "fits" bool);
CREATE FUNCTION "public"."st_boundingdiagonal"("geom" "public"."geometry", "fits" bool=false)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'ST_BoundingDiagonal'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."st_boundingdiagonal"("geom" "public"."geometry", "fits" bool) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_box2dfromgeohash
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_box2dfromgeohash"(text, int4);
CREATE FUNCTION "public"."st_box2dfromgeohash"(text, int4=NULL::integer)
  RETURNS "public"."box2d" AS '$libdir/postgis-3', 'box2d_from_geohash'
  LANGUAGE c IMMUTABLE
  COST 50;
ALTER FUNCTION "public"."st_box2dfromgeohash"(text, int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_buffer
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_buffer"(text, float8);
CREATE FUNCTION "public"."st_buffer"(text, float8)
  RETURNS "public"."geometry" AS $BODY$ SELECT public.ST_Buffer($1::public.geometry, $2);  $BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 100;
ALTER FUNCTION "public"."st_buffer"(text, float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_buffer
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_buffer"("public"."geography", float8, int4);
CREATE FUNCTION "public"."st_buffer"("public"."geography", float8, int4)
  RETURNS "public"."geography" AS $BODY$SELECT public.geography(public.ST_Transform(public.ST_Buffer(public.ST_Transform(public.geometry($1), public._ST_BestSRID($1)), $2, $3), public.ST_SRID($1)))$BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 100;
ALTER FUNCTION "public"."st_buffer"("public"."geography", float8, int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_buffer
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_buffer"("geom" "public"."geometry", "radius" float8, "quadsegs" int4);
CREATE FUNCTION "public"."st_buffer"("geom" "public"."geometry", "radius" float8, "quadsegs" int4)
  RETURNS "public"."geometry" AS $BODY$ SELECT public.ST_Buffer($1, $2, CAST('quad_segs='||CAST($3 AS text) as text)) $BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_buffer"("geom" "public"."geometry", "radius" float8, "quadsegs" int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_buffer
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_buffer"("geom" "public"."geometry", "radius" float8, "options" text);
CREATE FUNCTION "public"."st_buffer"("geom" "public"."geometry", "radius" float8, "options" text=''::text)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'buffer'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_buffer"("geom" "public"."geometry", "radius" float8, "options" text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_buffer
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_buffer"("public"."geography", float8);
CREATE FUNCTION "public"."st_buffer"("public"."geography", float8)
  RETURNS "public"."geography" AS $BODY$SELECT public.geography(public.ST_Transform(public.ST_Buffer(public.ST_Transform(public.geometry($1), public._ST_BestSRID($1)), $2), public.ST_SRID($1)))$BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 100;
ALTER FUNCTION "public"."st_buffer"("public"."geography", float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_buffer
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_buffer"(text, float8, text);
CREATE FUNCTION "public"."st_buffer"(text, float8, text)
  RETURNS "public"."geometry" AS $BODY$ SELECT public.ST_Buffer($1::public.geometry, $2, $3);  $BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 100;
ALTER FUNCTION "public"."st_buffer"(text, float8, text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_buffer
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_buffer"(text, float8, int4);
CREATE FUNCTION "public"."st_buffer"(text, float8, int4)
  RETURNS "public"."geometry" AS $BODY$ SELECT public.ST_Buffer($1::public.geometry, $2, $3);  $BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 100;
ALTER FUNCTION "public"."st_buffer"(text, float8, int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_buffer
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_buffer"("public"."geography", float8, text);
CREATE FUNCTION "public"."st_buffer"("public"."geography", float8, text)
  RETURNS "public"."geography" AS $BODY$SELECT public.geography(public.ST_Transform(public.ST_Buffer(public.ST_Transform(public.geometry($1), public._ST_BestSRID($1)), $2, $3), public.ST_SRID($1)))$BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 100;
ALTER FUNCTION "public"."st_buffer"("public"."geography", float8, text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_buildarea
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_buildarea"("public"."geometry");
CREATE FUNCTION "public"."st_buildarea"("public"."geometry")
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'ST_BuildArea'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_buildarea"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_centroid
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_centroid"("public"."geometry");
CREATE FUNCTION "public"."st_centroid"("public"."geometry")
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'centroid'
  LANGUAGE c IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_centroid"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_centroid
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_centroid"("public"."geography", "use_spheroid" bool);
CREATE FUNCTION "public"."st_centroid"("public"."geography", "use_spheroid" bool=true)
  RETURNS "public"."geography" AS '$libdir/postgis-3', 'geography_centroid'
  LANGUAGE c IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_centroid"("public"."geography", "use_spheroid" bool) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_centroid
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_centroid"(text);
CREATE FUNCTION "public"."st_centroid"(text)
  RETURNS "public"."geometry" AS $BODY$ SELECT public.ST_Centroid($1::public.geometry);  $BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 100;
ALTER FUNCTION "public"."st_centroid"(text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_chaikinsmoothing
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_chaikinsmoothing"("public"."geometry", int4, bool);
CREATE FUNCTION "public"."st_chaikinsmoothing"("public"."geometry", int4=1, bool=false)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'LWGEOM_ChaikinSmoothing'
  LANGUAGE c IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_chaikinsmoothing"("public"."geometry", int4, bool) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_cleangeometry
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_cleangeometry"("public"."geometry");
CREATE FUNCTION "public"."st_cleangeometry"("public"."geometry")
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'ST_CleanGeometry'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_cleangeometry"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_clipbybox2d
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_clipbybox2d"("geom" "public"."geometry", "box" "public"."box2d");
CREATE FUNCTION "public"."st_clipbybox2d"("geom" "public"."geometry", "box" "public"."box2d")
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'ST_ClipByBox2d'
  LANGUAGE c IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_clipbybox2d"("geom" "public"."geometry", "box" "public"."box2d") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_closestpoint
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_closestpoint"("public"."geography", "public"."geography", "use_spheroid" bool);
CREATE FUNCTION "public"."st_closestpoint"("public"."geography", "public"."geography", "use_spheroid" bool=true)
  RETURNS "public"."geography" AS '$libdir/postgis-3', 'geography_closestpoint'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."st_closestpoint"("public"."geography", "public"."geography", "use_spheroid" bool) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_closestpoint
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_closestpoint"("geom1" "public"."geometry", "geom2" "public"."geometry");
CREATE FUNCTION "public"."st_closestpoint"("geom1" "public"."geometry", "geom2" "public"."geometry")
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'LWGEOM_closestpoint'
  LANGUAGE c IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_closestpoint"("geom1" "public"."geometry", "geom2" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_closestpoint
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_closestpoint"(text, text);
CREATE FUNCTION "public"."st_closestpoint"(text, text)
  RETURNS "public"."geometry" AS $BODY$ SELECT public.ST_ClosestPoint($1::public.geometry, $2::public.geometry);  $BODY$
  LANGUAGE sql IMMUTABLE
  COST 100;
ALTER FUNCTION "public"."st_closestpoint"(text, text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_closestpointofapproach
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_closestpointofapproach"("public"."geometry", "public"."geometry");
CREATE FUNCTION "public"."st_closestpointofapproach"("public"."geometry", "public"."geometry")
  RETURNS "pg_catalog"."float8" AS '$libdir/postgis-3', 'ST_ClosestPointOfApproach'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_closestpointofapproach"("public"."geometry", "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_clusterdbscan
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_clusterdbscan"("public"."geometry", "eps" float8, "minpoints" int4);
CREATE FUNCTION "public"."st_clusterdbscan"("public"."geometry", "eps" float8, "minpoints" int4)
  RETURNS "pg_catalog"."int4" AS '$libdir/postgis-3', 'ST_ClusterDBSCAN'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_clusterdbscan"("public"."geometry", "eps" float8, "minpoints" int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_clusterintersecting
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_clusterintersecting"("public"."_geometry");
CREATE FUNCTION "public"."st_clusterintersecting"("public"."_geometry")
  RETURNS "public"."_geometry" AS '$libdir/postgis-3', 'clusterintersecting_garray'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_clusterintersecting"("public"."_geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_clusterintersectingwin
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_clusterintersectingwin"("public"."geometry");
CREATE FUNCTION "public"."st_clusterintersectingwin"("public"."geometry")
  RETURNS "pg_catalog"."int4" AS '$libdir/postgis-3', 'ST_ClusterIntersectingWin'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_clusterintersectingwin"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_clusterkmeans
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_clusterkmeans"("geom" "public"."geometry", "k" int4, "max_radius" float8);
CREATE FUNCTION "public"."st_clusterkmeans"("geom" "public"."geometry", "k" int4, "max_radius" float8=NULL::double precision)
  RETURNS "pg_catalog"."int4" AS '$libdir/postgis-3', 'ST_ClusterKMeans'
  LANGUAGE c VOLATILE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_clusterkmeans"("geom" "public"."geometry", "k" int4, "max_radius" float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_clusterwithin
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_clusterwithin"("public"."_geometry", float8);
CREATE FUNCTION "public"."st_clusterwithin"("public"."_geometry", float8)
  RETURNS "public"."_geometry" AS '$libdir/postgis-3', 'cluster_within_distance_garray'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_clusterwithin"("public"."_geometry", float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_clusterwithinwin
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_clusterwithinwin"("public"."geometry", "distance" float8);
CREATE FUNCTION "public"."st_clusterwithinwin"("public"."geometry", "distance" float8)
  RETURNS "pg_catalog"."int4" AS '$libdir/postgis-3', 'ST_ClusterWithinWin'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_clusterwithinwin"("public"."geometry", "distance" float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_collect
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_collect"("geom1" "public"."geometry", "geom2" "public"."geometry");
CREATE FUNCTION "public"."st_collect"("geom1" "public"."geometry", "geom2" "public"."geometry")
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'LWGEOM_collect'
  LANGUAGE c IMMUTABLE
  COST 50;
ALTER FUNCTION "public"."st_collect"("geom1" "public"."geometry", "geom2" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_collect
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_collect"("public"."_geometry");
CREATE FUNCTION "public"."st_collect"("public"."_geometry")
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'LWGEOM_collect_garray'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_collect"("public"."_geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_collectionextract
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_collectionextract"("public"."geometry");
CREATE FUNCTION "public"."st_collectionextract"("public"."geometry")
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'ST_CollectionExtract'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_collectionextract"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_collectionextract
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_collectionextract"("public"."geometry", int4);
CREATE FUNCTION "public"."st_collectionextract"("public"."geometry", int4)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'ST_CollectionExtract'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_collectionextract"("public"."geometry", int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_collectionhomogenize
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_collectionhomogenize"("public"."geometry");
CREATE FUNCTION "public"."st_collectionhomogenize"("public"."geometry")
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'ST_CollectionHomogenize'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_collectionhomogenize"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_combinebbox
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_combinebbox"("public"."box3d", "public"."box3d");
CREATE FUNCTION "public"."st_combinebbox"("public"."box3d", "public"."box3d")
  RETURNS "public"."box3d" AS '$libdir/postgis-3', 'BOX3D_combine_BOX3D'
  LANGUAGE c IMMUTABLE
  COST 50;
ALTER FUNCTION "public"."st_combinebbox"("public"."box3d", "public"."box3d") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_combinebbox
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_combinebbox"("public"."box3d", "public"."geometry");
CREATE FUNCTION "public"."st_combinebbox"("public"."box3d", "public"."geometry")
  RETURNS "public"."box3d" AS '$libdir/postgis-3', 'BOX3D_combine'
  LANGUAGE c IMMUTABLE
  COST 50;
ALTER FUNCTION "public"."st_combinebbox"("public"."box3d", "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_combinebbox
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_combinebbox"("public"."box2d", "public"."geometry");
CREATE FUNCTION "public"."st_combinebbox"("public"."box2d", "public"."geometry")
  RETURNS "public"."box2d" AS '$libdir/postgis-3', 'BOX2D_combine'
  LANGUAGE c IMMUTABLE
  COST 1;
ALTER FUNCTION "public"."st_combinebbox"("public"."box2d", "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_concavehull
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_concavehull"("param_geom" "public"."geometry", "param_pctconvex" float8, "param_allow_holes" bool);
CREATE FUNCTION "public"."st_concavehull"("param_geom" "public"."geometry", "param_pctconvex" float8, "param_allow_holes" bool=false)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'ST_ConcaveHull'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_concavehull"("param_geom" "public"."geometry", "param_pctconvex" float8, "param_allow_holes" bool) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_contains
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_contains"("geom1" "public"."geometry", "geom2" "public"."geometry");
CREATE FUNCTION "public"."st_contains"("geom1" "public"."geometry", "geom2" "public"."geometry")
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'contains'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_contains"("geom1" "public"."geometry", "geom2" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_containsproperly
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_containsproperly"("geom1" "public"."geometry", "geom2" "public"."geometry");
CREATE FUNCTION "public"."st_containsproperly"("geom1" "public"."geometry", "geom2" "public"."geometry")
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'containsproperly'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_containsproperly"("geom1" "public"."geometry", "geom2" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_convexhull
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_convexhull"("public"."geometry");
CREATE FUNCTION "public"."st_convexhull"("public"."geometry")
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'convexhull'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_convexhull"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_coorddim
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_coorddim"("geometry" "public"."geometry");
CREATE FUNCTION "public"."st_coorddim"("geometry" "public"."geometry")
  RETURNS "pg_catalog"."int2" AS '$libdir/postgis-3', 'LWGEOM_ndims'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."st_coorddim"("geometry" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_coverageclean
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_coverageclean"("geom" "public"."geometry", "gapmaximumwidth" float8, "snappingdistance" float8, "overlapmergestrategy" text);
CREATE FUNCTION "public"."st_coverageclean"("geom" "public"."geometry", "gapmaximumwidth" float8=0.0, "snappingdistance" float8='-1.0'::numeric, "overlapmergestrategy" text='MERGE_LONGEST_BORDER'::text)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'ST_CoverageClean'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_coverageclean"("geom" "public"."geometry", "gapmaximumwidth" float8, "snappingdistance" float8, "overlapmergestrategy" text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_coverageinvalidedges
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_coverageinvalidedges"("geom" "public"."geometry", "tolerance" float8);
CREATE FUNCTION "public"."st_coverageinvalidedges"("geom" "public"."geometry", "tolerance" float8=0.0)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'ST_CoverageInvalidEdges'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_coverageinvalidedges"("geom" "public"."geometry", "tolerance" float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_coveragesimplify
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_coveragesimplify"("geom" "public"."geometry", "tolerance" float8, "simplifyboundary" bool);
CREATE FUNCTION "public"."st_coveragesimplify"("geom" "public"."geometry", "tolerance" float8, "simplifyboundary" bool=true)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'ST_CoverageSimplify'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_coveragesimplify"("geom" "public"."geometry", "tolerance" float8, "simplifyboundary" bool) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_coverageunion
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_coverageunion"("public"."_geometry");
CREATE FUNCTION "public"."st_coverageunion"("public"."_geometry")
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'ST_CoverageUnion'
  LANGUAGE c IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_coverageunion"("public"."_geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_coveredby
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_coveredby"(text, text);
CREATE FUNCTION "public"."st_coveredby"(text, text)
  RETURNS "pg_catalog"."bool" AS $BODY$ SELECT public.ST_CoveredBy($1::public.geometry, $2::public.geometry);  $BODY$
  LANGUAGE sql IMMUTABLE
  COST 100;
ALTER FUNCTION "public"."st_coveredby"(text, text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_coveredby
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_coveredby"("geom1" "public"."geometry", "geom2" "public"."geometry");
CREATE FUNCTION "public"."st_coveredby"("geom1" "public"."geometry", "geom2" "public"."geometry")
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'coveredby'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_coveredby"("geom1" "public"."geometry", "geom2" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_coveredby
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_coveredby"("geog1" "public"."geography", "geog2" "public"."geography");
CREATE FUNCTION "public"."st_coveredby"("geog1" "public"."geography", "geog2" "public"."geography")
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'geography_coveredby'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_coveredby"("geog1" "public"."geography", "geog2" "public"."geography") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_covers
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_covers"(text, text);
CREATE FUNCTION "public"."st_covers"(text, text)
  RETURNS "pg_catalog"."bool" AS $BODY$ SELECT public.ST_Covers($1::public.geometry, $2::public.geometry);  $BODY$
  LANGUAGE sql IMMUTABLE
  COST 100;
ALTER FUNCTION "public"."st_covers"(text, text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_covers
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_covers"("geom1" "public"."geometry", "geom2" "public"."geometry");
CREATE FUNCTION "public"."st_covers"("geom1" "public"."geometry", "geom2" "public"."geometry")
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'covers'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_covers"("geom1" "public"."geometry", "geom2" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_covers
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_covers"("geog1" "public"."geography", "geog2" "public"."geography");
CREATE FUNCTION "public"."st_covers"("geog1" "public"."geography", "geog2" "public"."geography")
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'geography_covers'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_covers"("geog1" "public"."geography", "geog2" "public"."geography") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_cpawithin
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_cpawithin"("public"."geometry", "public"."geometry", float8);
CREATE FUNCTION "public"."st_cpawithin"("public"."geometry", "public"."geometry", float8)
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'ST_CPAWithin'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_cpawithin"("public"."geometry", "public"."geometry", float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_crosses
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_crosses"("geom1" "public"."geometry", "geom2" "public"."geometry");
CREATE FUNCTION "public"."st_crosses"("geom1" "public"."geometry", "geom2" "public"."geometry")
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'crosses'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_crosses"("geom1" "public"."geometry", "geom2" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_curven
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_curven"("geometry" "public"."geometry", "i" int4);
CREATE FUNCTION "public"."st_curven"("geometry" "public"."geometry", "i" int4)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'ST_CurveN'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_curven"("geometry" "public"."geometry", "i" int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_curvetoline
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_curvetoline"("geom" "public"."geometry", "tol" float8, "toltype" int4, "flags" int4);
CREATE FUNCTION "public"."st_curvetoline"("geom" "public"."geometry", "tol" float8=32, "toltype" int4=0, "flags" int4=0)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'ST_CurveToLine'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_curvetoline"("geom" "public"."geometry", "tol" float8, "toltype" int4, "flags" int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_delaunaytriangles
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_delaunaytriangles"("g1" "public"."geometry", "tolerance" float8, "flags" int4);
CREATE FUNCTION "public"."st_delaunaytriangles"("g1" "public"."geometry", "tolerance" float8=0.0, "flags" int4=0)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'ST_DelaunayTriangles'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_delaunaytriangles"("g1" "public"."geometry", "tolerance" float8, "flags" int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_dfullywithin
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_dfullywithin"("geom1" "public"."geometry", "geom2" "public"."geometry", float8);
CREATE FUNCTION "public"."st_dfullywithin"("geom1" "public"."geometry", "geom2" "public"."geometry", float8)
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'LWGEOM_dfullywithin'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_dfullywithin"("geom1" "public"."geometry", "geom2" "public"."geometry", float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_difference
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_difference"("geom1" "public"."geometry", "geom2" "public"."geometry", "gridsize" float8);
CREATE FUNCTION "public"."st_difference"("geom1" "public"."geometry", "geom2" "public"."geometry", "gridsize" float8='-1.0'::numeric)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'ST_Difference'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_difference"("geom1" "public"."geometry", "geom2" "public"."geometry", "gridsize" float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_dimension
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_dimension"("public"."geometry");
CREATE FUNCTION "public"."st_dimension"("public"."geometry")
  RETURNS "pg_catalog"."int4" AS '$libdir/postgis-3', 'LWGEOM_dimension'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_dimension"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_disjoint
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_disjoint"("geom1" "public"."geometry", "geom2" "public"."geometry");
CREATE FUNCTION "public"."st_disjoint"("geom1" "public"."geometry", "geom2" "public"."geometry")
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'disjoint'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_disjoint"("geom1" "public"."geometry", "geom2" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_distance
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_distance"("geog1" "public"."geography", "geog2" "public"."geography", "use_spheroid" bool);
CREATE FUNCTION "public"."st_distance"("geog1" "public"."geography", "geog2" "public"."geography", "use_spheroid" bool=true)
  RETURNS "pg_catalog"."float8" AS '$libdir/postgis-3', 'geography_distance'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_distance"("geog1" "public"."geography", "geog2" "public"."geography", "use_spheroid" bool) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_distance
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_distance"(text, text);
CREATE FUNCTION "public"."st_distance"(text, text)
  RETURNS "pg_catalog"."float8" AS $BODY$ SELECT public.ST_Distance($1::public.geometry, $2::public.geometry);  $BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 100;
ALTER FUNCTION "public"."st_distance"(text, text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_distance
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_distance"("geom1" "public"."geometry", "geom2" "public"."geometry");
CREATE FUNCTION "public"."st_distance"("geom1" "public"."geometry", "geom2" "public"."geometry")
  RETURNS "pg_catalog"."float8" AS '$libdir/postgis-3', 'ST_Distance'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_distance"("geom1" "public"."geometry", "geom2" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_distancecpa
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_distancecpa"("public"."geometry", "public"."geometry");
CREATE FUNCTION "public"."st_distancecpa"("public"."geometry", "public"."geometry")
  RETURNS "pg_catalog"."float8" AS '$libdir/postgis-3', 'ST_DistanceCPA'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_distancecpa"("public"."geometry", "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_distancesphere
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_distancesphere"("geom1" "public"."geometry", "geom2" "public"."geometry");
CREATE FUNCTION "public"."st_distancesphere"("geom1" "public"."geometry", "geom2" "public"."geometry")
  RETURNS "pg_catalog"."float8" AS $BODY$select public.ST_distance( public.geography($1), public.geography($2),false)$BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 100;
ALTER FUNCTION "public"."st_distancesphere"("geom1" "public"."geometry", "geom2" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_distancesphere
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_distancesphere"("geom1" "public"."geometry", "geom2" "public"."geometry", "radius" float8);
CREATE FUNCTION "public"."st_distancesphere"("geom1" "public"."geometry", "geom2" "public"."geometry", "radius" float8)
  RETURNS "pg_catalog"."float8" AS '$libdir/postgis-3', 'LWGEOM_distance_sphere'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_distancesphere"("geom1" "public"."geometry", "geom2" "public"."geometry", "radius" float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_distancespheroid
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_distancespheroid"("geom1" "public"."geometry", "geom2" "public"."geometry");
CREATE FUNCTION "public"."st_distancespheroid"("geom1" "public"."geometry", "geom2" "public"."geometry")
  RETURNS "pg_catalog"."float8" AS '$libdir/postgis-3', 'LWGEOM_distance_ellipsoid'
  LANGUAGE c IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_distancespheroid"("geom1" "public"."geometry", "geom2" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_distancespheroid
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_distancespheroid"("geom1" "public"."geometry", "geom2" "public"."geometry", "public"."spheroid");
CREATE FUNCTION "public"."st_distancespheroid"("geom1" "public"."geometry", "geom2" "public"."geometry", "public"."spheroid")
  RETURNS "pg_catalog"."float8" AS '$libdir/postgis-3', 'LWGEOM_distance_ellipsoid'
  LANGUAGE c IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_distancespheroid"("geom1" "public"."geometry", "geom2" "public"."geometry", "public"."spheroid") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_dump
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_dump"("public"."geometry");
CREATE FUNCTION "public"."st_dump"("public"."geometry")
  RETURNS SETOF "public"."geometry_dump" AS '$libdir/postgis-3', 'LWGEOM_dump'
  LANGUAGE c IMMUTABLE STRICT
  COST 250
  ROWS 1000;
ALTER FUNCTION "public"."st_dump"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_dumppoints
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_dumppoints"("public"."geometry");
CREATE FUNCTION "public"."st_dumppoints"("public"."geometry")
  RETURNS SETOF "public"."geometry_dump" AS '$libdir/postgis-3', 'LWGEOM_dumppoints'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000
  ROWS 1000;
ALTER FUNCTION "public"."st_dumppoints"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_dumprings
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_dumprings"("public"."geometry");
CREATE FUNCTION "public"."st_dumprings"("public"."geometry")
  RETURNS SETOF "public"."geometry_dump" AS '$libdir/postgis-3', 'LWGEOM_dump_rings'
  LANGUAGE c IMMUTABLE STRICT
  COST 250
  ROWS 1000;
ALTER FUNCTION "public"."st_dumprings"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_dumpsegments
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_dumpsegments"("public"."geometry");
CREATE FUNCTION "public"."st_dumpsegments"("public"."geometry")
  RETURNS SETOF "public"."geometry_dump" AS '$libdir/postgis-3', 'LWGEOM_dumpsegments'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000
  ROWS 1000;
ALTER FUNCTION "public"."st_dumpsegments"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_dwithin
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_dwithin"("geom1" "public"."geometry", "geom2" "public"."geometry", float8);
CREATE FUNCTION "public"."st_dwithin"("geom1" "public"."geometry", "geom2" "public"."geometry", float8)
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'LWGEOM_dwithin'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_dwithin"("geom1" "public"."geometry", "geom2" "public"."geometry", float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_dwithin
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_dwithin"("geog1" "public"."geography", "geog2" "public"."geography", "tolerance" float8, "use_spheroid" bool);
CREATE FUNCTION "public"."st_dwithin"("geog1" "public"."geography", "geog2" "public"."geography", "tolerance" float8, "use_spheroid" bool=true)
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'geography_dwithin'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_dwithin"("geog1" "public"."geography", "geog2" "public"."geography", "tolerance" float8, "use_spheroid" bool) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_dwithin
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_dwithin"(text, text, float8);
CREATE FUNCTION "public"."st_dwithin"(text, text, float8)
  RETURNS "pg_catalog"."bool" AS $BODY$ SELECT public.ST_DWithin($1::public.geometry, $2::public.geometry, $3);  $BODY$
  LANGUAGE sql IMMUTABLE
  COST 100;
ALTER FUNCTION "public"."st_dwithin"(text, text, float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_endpoint
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_endpoint"("public"."geometry");
CREATE FUNCTION "public"."st_endpoint"("public"."geometry")
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'LWGEOM_endpoint_linestring'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_endpoint"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_envelope
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_envelope"("public"."geometry");
CREATE FUNCTION "public"."st_envelope"("public"."geometry")
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'LWGEOM_envelope'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_envelope"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_equals
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_equals"("geom1" "public"."geometry", "geom2" "public"."geometry");
CREATE FUNCTION "public"."st_equals"("geom1" "public"."geometry", "geom2" "public"."geometry")
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'ST_Equals'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_equals"("geom1" "public"."geometry", "geom2" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_estimatedextent
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_estimatedextent"(text, text);
CREATE FUNCTION "public"."st_estimatedextent"(text, text)
  RETURNS "public"."box2d" AS '$libdir/postgis-3', 'gserialized_estimated_extent'
  LANGUAGE c STABLE STRICT
  COST 1;
ALTER FUNCTION "public"."st_estimatedextent"(text, text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_estimatedextent
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_estimatedextent"(text, text, text, bool);
CREATE FUNCTION "public"."st_estimatedextent"(text, text, text, bool)
  RETURNS "public"."box2d" AS '$libdir/postgis-3', 'gserialized_estimated_extent'
  LANGUAGE c STABLE STRICT
  COST 1;
ALTER FUNCTION "public"."st_estimatedextent"(text, text, text, bool) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_estimatedextent
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_estimatedextent"(text, text, text);
CREATE FUNCTION "public"."st_estimatedextent"(text, text, text)
  RETURNS "public"."box2d" AS '$libdir/postgis-3', 'gserialized_estimated_extent'
  LANGUAGE c STABLE STRICT
  COST 1;
ALTER FUNCTION "public"."st_estimatedextent"(text, text, text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_expand
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_expand"("public"."geometry", float8);
CREATE FUNCTION "public"."st_expand"("public"."geometry", float8)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'LWGEOM_expand'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_expand"("public"."geometry", float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_expand
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_expand"("geom" "public"."geometry", "dx" float8, "dy" float8, "dz" float8, "dm" float8);
CREATE FUNCTION "public"."st_expand"("geom" "public"."geometry", "dx" float8, "dy" float8, "dz" float8=0, "dm" float8=0)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'LWGEOM_expand'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_expand"("geom" "public"."geometry", "dx" float8, "dy" float8, "dz" float8, "dm" float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_expand
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_expand"("box" "public"."box2d", "dx" float8, "dy" float8);
CREATE FUNCTION "public"."st_expand"("box" "public"."box2d", "dx" float8, "dy" float8)
  RETURNS "public"."box2d" AS '$libdir/postgis-3', 'BOX2D_expand'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."st_expand"("box" "public"."box2d", "dx" float8, "dy" float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_expand
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_expand"("public"."box3d", float8);
CREATE FUNCTION "public"."st_expand"("public"."box3d", float8)
  RETURNS "public"."box3d" AS '$libdir/postgis-3', 'BOX3D_expand'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_expand"("public"."box3d", float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_expand
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_expand"("public"."box2d", float8);
CREATE FUNCTION "public"."st_expand"("public"."box2d", float8)
  RETURNS "public"."box2d" AS '$libdir/postgis-3', 'BOX2D_expand'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."st_expand"("public"."box2d", float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_expand
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_expand"("box" "public"."box3d", "dx" float8, "dy" float8, "dz" float8);
CREATE FUNCTION "public"."st_expand"("box" "public"."box3d", "dx" float8, "dy" float8, "dz" float8=0)
  RETURNS "public"."box3d" AS '$libdir/postgis-3', 'BOX3D_expand'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_expand"("box" "public"."box3d", "dx" float8, "dy" float8, "dz" float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_exteriorring
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_exteriorring"("public"."geometry");
CREATE FUNCTION "public"."st_exteriorring"("public"."geometry")
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'LWGEOM_exteriorring_polygon'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_exteriorring"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_filterbym
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_filterbym"("public"."geometry", float8, float8, bool);
CREATE FUNCTION "public"."st_filterbym"("public"."geometry", float8, float8=NULL::double precision, bool=false)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'LWGEOM_FilterByM'
  LANGUAGE c IMMUTABLE
  COST 50;
ALTER FUNCTION "public"."st_filterbym"("public"."geometry", float8, float8, bool) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_findextent
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_findextent"(text, text);
CREATE FUNCTION "public"."st_findextent"(text, text)
  RETURNS "public"."box2d" AS $BODY$
DECLARE
	tablename alias for $1;
	columnname alias for $2;
	myrec RECORD;

BEGIN
	FOR myrec IN EXECUTE 'SELECT public.ST_Extent("' || columnname || '") As extent FROM "' || tablename || '"' LOOP
		return myrec.extent;
	END LOOP;
END;
$BODY$
  LANGUAGE plpgsql STABLE STRICT
  COST 100;
ALTER FUNCTION "public"."st_findextent"(text, text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_findextent
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_findextent"(text, text, text);
CREATE FUNCTION "public"."st_findextent"(text, text, text)
  RETURNS "public"."box2d" AS $BODY$
DECLARE
	schemaname alias for $1;
	tablename alias for $2;
	columnname alias for $3;
	myrec RECORD;
BEGIN
	FOR myrec IN EXECUTE 'SELECT public.ST_Extent("' || columnname || '") As extent FROM "' || schemaname || '"."' || tablename || '"' LOOP
		return myrec.extent;
	END LOOP;
END;
$BODY$
  LANGUAGE plpgsql STABLE STRICT
  COST 100;
ALTER FUNCTION "public"."st_findextent"(text, text, text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_flipcoordinates
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_flipcoordinates"("public"."geometry");
CREATE FUNCTION "public"."st_flipcoordinates"("public"."geometry")
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'ST_FlipCoordinates'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_flipcoordinates"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_force2d
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_force2d"("public"."geometry");
CREATE FUNCTION "public"."st_force2d"("public"."geometry")
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'LWGEOM_force_2d'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_force2d"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_force3d
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_force3d"("geom" "public"."geometry", "zvalue" float8);
CREATE FUNCTION "public"."st_force3d"("geom" "public"."geometry", "zvalue" float8=0.0)
  RETURNS "public"."geometry" AS $BODY$SELECT public.ST_Force3DZ($1, $2)$BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_force3d"("geom" "public"."geometry", "zvalue" float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_force3dm
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_force3dm"("geom" "public"."geometry", "mvalue" float8);
CREATE FUNCTION "public"."st_force3dm"("geom" "public"."geometry", "mvalue" float8=0.0)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'LWGEOM_force_3dm'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_force3dm"("geom" "public"."geometry", "mvalue" float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_force3dz
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_force3dz"("geom" "public"."geometry", "zvalue" float8);
CREATE FUNCTION "public"."st_force3dz"("geom" "public"."geometry", "zvalue" float8=0.0)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'LWGEOM_force_3dz'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_force3dz"("geom" "public"."geometry", "zvalue" float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_force4d
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_force4d"("geom" "public"."geometry", "zvalue" float8, "mvalue" float8);
CREATE FUNCTION "public"."st_force4d"("geom" "public"."geometry", "zvalue" float8=0.0, "mvalue" float8=0.0)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'LWGEOM_force_4d'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_force4d"("geom" "public"."geometry", "zvalue" float8, "mvalue" float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_forcecollection
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_forcecollection"("public"."geometry");
CREATE FUNCTION "public"."st_forcecollection"("public"."geometry")
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'LWGEOM_force_collection'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_forcecollection"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_forcecurve
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_forcecurve"("public"."geometry");
CREATE FUNCTION "public"."st_forcecurve"("public"."geometry")
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'LWGEOM_force_curve'
  LANGUAGE c IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_forcecurve"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_forcepolygonccw
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_forcepolygonccw"("public"."geometry");
CREATE FUNCTION "public"."st_forcepolygonccw"("public"."geometry")
  RETURNS "public"."geometry" AS $BODY$ SELECT public.ST_Reverse(public.ST_ForcePolygonCW($1)) $BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_forcepolygonccw"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_forcepolygoncw
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_forcepolygoncw"("public"."geometry");
CREATE FUNCTION "public"."st_forcepolygoncw"("public"."geometry")
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'LWGEOM_force_clockwise_poly'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_forcepolygoncw"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_forcerhr
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_forcerhr"("public"."geometry");
CREATE FUNCTION "public"."st_forcerhr"("public"."geometry")
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'LWGEOM_force_clockwise_poly'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_forcerhr"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_forcesfs
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_forcesfs"("public"."geometry", "version" text);
CREATE FUNCTION "public"."st_forcesfs"("public"."geometry", "version" text)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'LWGEOM_force_sfs'
  LANGUAGE c IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_forcesfs"("public"."geometry", "version" text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_forcesfs
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_forcesfs"("public"."geometry");
CREATE FUNCTION "public"."st_forcesfs"("public"."geometry")
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'LWGEOM_force_sfs'
  LANGUAGE c IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_forcesfs"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_frechetdistance
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_frechetdistance"("geom1" "public"."geometry", "geom2" "public"."geometry", float8);
CREATE FUNCTION "public"."st_frechetdistance"("geom1" "public"."geometry", "geom2" "public"."geometry", float8='-1'::integer)
  RETURNS "pg_catalog"."float8" AS '$libdir/postgis-3', 'ST_FrechetDistance'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_frechetdistance"("geom1" "public"."geometry", "geom2" "public"."geometry", float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_fromflatgeobuf
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_fromflatgeobuf"(anyelement, bytea);
CREATE FUNCTION "public"."st_fromflatgeobuf"(anyelement, bytea)
  RETURNS SETOF "pg_catalog"."anyelement" AS '$libdir/postgis-3', 'pgis_fromflatgeobuf'
  LANGUAGE c IMMUTABLE
  COST 250
  ROWS 1000;
ALTER FUNCTION "public"."st_fromflatgeobuf"(anyelement, bytea) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_fromflatgeobuftotable
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_fromflatgeobuftotable"(text, text, bytea);
CREATE FUNCTION "public"."st_fromflatgeobuftotable"(text, text, bytea)
  RETURNS "pg_catalog"."void" AS '$libdir/postgis-3', 'pgis_tablefromflatgeobuf'
  LANGUAGE c IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_fromflatgeobuftotable"(text, text, bytea) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_generatepoints
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_generatepoints"("area" "public"."geometry", "npoints" int4);
CREATE FUNCTION "public"."st_generatepoints"("area" "public"."geometry", "npoints" int4)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'ST_GeneratePoints'
  LANGUAGE c VOLATILE STRICT
  COST 250;
ALTER FUNCTION "public"."st_generatepoints"("area" "public"."geometry", "npoints" int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_generatepoints
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_generatepoints"("area" "public"."geometry", "npoints" int4, "seed" int4);
CREATE FUNCTION "public"."st_generatepoints"("area" "public"."geometry", "npoints" int4, "seed" int4)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'ST_GeneratePoints'
  LANGUAGE c IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_generatepoints"("area" "public"."geometry", "npoints" int4, "seed" int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_geogfromtext
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_geogfromtext"(text);
CREATE FUNCTION "public"."st_geogfromtext"(text)
  RETURNS "public"."geography" AS '$libdir/postgis-3', 'geography_from_text'
  LANGUAGE c IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_geogfromtext"(text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_geogfromwkb
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_geogfromwkb"(bytea);
CREATE FUNCTION "public"."st_geogfromwkb"(bytea)
  RETURNS "public"."geography" AS '$libdir/postgis-3', 'geography_from_binary'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_geogfromwkb"(bytea) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_geographyfromtext
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_geographyfromtext"(text);
CREATE FUNCTION "public"."st_geographyfromtext"(text)
  RETURNS "public"."geography" AS '$libdir/postgis-3', 'geography_from_text'
  LANGUAGE c IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_geographyfromtext"(text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_geohash
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_geohash"("geog" "public"."geography", "maxchars" int4);
CREATE FUNCTION "public"."st_geohash"("geog" "public"."geography", "maxchars" int4=0)
  RETURNS "pg_catalog"."text" AS '$libdir/postgis-3', 'ST_GeoHash'
  LANGUAGE c IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_geohash"("geog" "public"."geography", "maxchars" int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_geohash
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_geohash"("geom" "public"."geometry", "maxchars" int4);
CREATE FUNCTION "public"."st_geohash"("geom" "public"."geometry", "maxchars" int4=0)
  RETURNS "pg_catalog"."text" AS '$libdir/postgis-3', 'ST_GeoHash'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_geohash"("geom" "public"."geometry", "maxchars" int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_geomcollfromtext
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_geomcollfromtext"(text);
CREATE FUNCTION "public"."st_geomcollfromtext"(text)
  RETURNS "public"."geometry" AS $BODY$
	SELECT CASE
	WHEN public.ST_GeometryType(public.ST_GeomFromText($1)) = 'ST_GeometryCollection'
	THEN public.ST_GeomFromText($1)
	ELSE NULL END
	$BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_geomcollfromtext"(text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_geomcollfromtext
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_geomcollfromtext"(text, int4);
CREATE FUNCTION "public"."st_geomcollfromtext"(text, int4)
  RETURNS "public"."geometry" AS $BODY$
	SELECT CASE
	WHEN public.ST_GeometryType(public.ST_GeomFromText($1, $2)) = 'ST_GeometryCollection'
	THEN public.ST_GeomFromText($1,$2)
	ELSE NULL END
	$BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_geomcollfromtext"(text, int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_geomcollfromwkb
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_geomcollfromwkb"(bytea, int4);
CREATE FUNCTION "public"."st_geomcollfromwkb"(bytea, int4)
  RETURNS "public"."geometry" AS $BODY$
	SELECT CASE
	WHEN public.ST_GeometryType(public.ST_GeomFromWKB($1, $2)) = 'ST_GeometryCollection'
	THEN public.ST_GeomFromWKB($1, $2)
	ELSE NULL END
	$BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_geomcollfromwkb"(bytea, int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_geomcollfromwkb
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_geomcollfromwkb"(bytea);
CREATE FUNCTION "public"."st_geomcollfromwkb"(bytea)
  RETURNS "public"."geometry" AS $BODY$
	SELECT CASE
	WHEN public.ST_GeometryType(public.ST_GeomFromWKB($1)) = 'ST_GeometryCollection'
	THEN public.ST_GeomFromWKB($1)
	ELSE NULL END
	$BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_geomcollfromwkb"(bytea) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_geometricmedian
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_geometricmedian"("g" "public"."geometry", "tolerance" float8, "max_iter" int4, "fail_if_not_converged" bool);
CREATE FUNCTION "public"."st_geometricmedian"("g" "public"."geometry", "tolerance" float8=NULL::double precision, "max_iter" int4=10000, "fail_if_not_converged" bool=false)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'ST_GeometricMedian'
  LANGUAGE c IMMUTABLE
  COST 5000;
ALTER FUNCTION "public"."st_geometricmedian"("g" "public"."geometry", "tolerance" float8, "max_iter" int4, "fail_if_not_converged" bool) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_geometryfromtext
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_geometryfromtext"(text, int4);
CREATE FUNCTION "public"."st_geometryfromtext"(text, int4)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'LWGEOM_from_text'
  LANGUAGE c IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_geometryfromtext"(text, int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_geometryfromtext
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_geometryfromtext"(text);
CREATE FUNCTION "public"."st_geometryfromtext"(text)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'LWGEOM_from_text'
  LANGUAGE c IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_geometryfromtext"(text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_geometryn
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_geometryn"("public"."geometry", int4);
CREATE FUNCTION "public"."st_geometryn"("public"."geometry", int4)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'LWGEOM_geometryn_collection'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_geometryn"("public"."geometry", int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_geometrytype
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_geometrytype"("public"."geometry");
CREATE FUNCTION "public"."st_geometrytype"("public"."geometry")
  RETURNS "pg_catalog"."text" AS '$libdir/postgis-3', 'geometry_geometrytype'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."st_geometrytype"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_geomfromewkb
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_geomfromewkb"(bytea);
CREATE FUNCTION "public"."st_geomfromewkb"(bytea)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'LWGEOMFromEWKB'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_geomfromewkb"(bytea) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_geomfromewkt
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_geomfromewkt"(text);
CREATE FUNCTION "public"."st_geomfromewkt"(text)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'parse_WKT_lwgeom'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_geomfromewkt"(text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_geomfromgeohash
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_geomfromgeohash"(text, int4);
CREATE FUNCTION "public"."st_geomfromgeohash"(text, int4=NULL::integer)
  RETURNS "public"."geometry" AS $BODY$ SELECT CAST(public.ST_Box2dFromGeoHash($1, $2) AS public.geometry); $BODY$
  LANGUAGE sql IMMUTABLE
  COST 50;
ALTER FUNCTION "public"."st_geomfromgeohash"(text, int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_geomfromgeojson
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_geomfromgeojson"(text);
CREATE FUNCTION "public"."st_geomfromgeojson"(text)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'geom_from_geojson'
  LANGUAGE c IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_geomfromgeojson"(text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_geomfromgeojson
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_geomfromgeojson"(jsonb);
CREATE FUNCTION "public"."st_geomfromgeojson"(jsonb)
  RETURNS "public"."geometry" AS $BODY$SELECT public.ST_GeomFromGeoJson($1::text)$BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_geomfromgeojson"(jsonb) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_geomfromgeojson
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_geomfromgeojson"(json);
CREATE FUNCTION "public"."st_geomfromgeojson"(json)
  RETURNS "public"."geometry" AS $BODY$SELECT public.ST_GeomFromGeoJson($1::text)$BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_geomfromgeojson"(json) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_geomfromgml
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_geomfromgml"(text, int4);
CREATE FUNCTION "public"."st_geomfromgml"(text, int4)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'geom_from_gml'
  LANGUAGE c IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_geomfromgml"(text, int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_geomfromgml
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_geomfromgml"(text);
CREATE FUNCTION "public"."st_geomfromgml"(text)
  RETURNS "public"."geometry" AS $BODY$SELECT public._ST_GeomFromGML($1, 0)$BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_geomfromgml"(text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_geomfromkml
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_geomfromkml"(text);
CREATE FUNCTION "public"."st_geomfromkml"(text)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'geom_from_kml'
  LANGUAGE c IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_geomfromkml"(text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_geomfrommarc21
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_geomfrommarc21"("marc21xml" text);
CREATE FUNCTION "public"."st_geomfrommarc21"("marc21xml" text)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'ST_GeomFromMARC21'
  LANGUAGE c IMMUTABLE STRICT
  COST 500;
ALTER FUNCTION "public"."st_geomfrommarc21"("marc21xml" text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_geomfromtext
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_geomfromtext"(text, int4);
CREATE FUNCTION "public"."st_geomfromtext"(text, int4)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'LWGEOM_from_text'
  LANGUAGE c IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_geomfromtext"(text, int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_geomfromtext
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_geomfromtext"(text);
CREATE FUNCTION "public"."st_geomfromtext"(text)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'LWGEOM_from_text'
  LANGUAGE c IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_geomfromtext"(text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_geomfromtwkb
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_geomfromtwkb"(bytea);
CREATE FUNCTION "public"."st_geomfromtwkb"(bytea)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'LWGEOMFromTWKB'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_geomfromtwkb"(bytea) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_geomfromwkb
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_geomfromwkb"(bytea, int4);
CREATE FUNCTION "public"."st_geomfromwkb"(bytea, int4)
  RETURNS "public"."geometry" AS $BODY$SELECT public.ST_SetSRID(public.ST_GeomFromWKB($1), $2)$BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_geomfromwkb"(bytea, int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_geomfromwkb
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_geomfromwkb"(bytea);
CREATE FUNCTION "public"."st_geomfromwkb"(bytea)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'LWGEOM_from_WKB'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_geomfromwkb"(bytea) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_gmltosql
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_gmltosql"(text);
CREATE FUNCTION "public"."st_gmltosql"(text)
  RETURNS "public"."geometry" AS $BODY$SELECT public._ST_GeomFromGML($1, 0)$BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_gmltosql"(text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_gmltosql
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_gmltosql"(text, int4);
CREATE FUNCTION "public"."st_gmltosql"(text, int4)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'geom_from_gml'
  LANGUAGE c IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_gmltosql"(text, int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_hasarc
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_hasarc"("geometry" "public"."geometry");
CREATE FUNCTION "public"."st_hasarc"("geometry" "public"."geometry")
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'LWGEOM_has_arc'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_hasarc"("geometry" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_hasm
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_hasm"("public"."geometry");
CREATE FUNCTION "public"."st_hasm"("public"."geometry")
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'LWGEOM_hasm'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."st_hasm"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_hasz
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_hasz"("public"."geometry");
CREATE FUNCTION "public"."st_hasz"("public"."geometry")
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'LWGEOM_hasz'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."st_hasz"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_hausdorffdistance
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_hausdorffdistance"("geom1" "public"."geometry", "geom2" "public"."geometry", float8);
CREATE FUNCTION "public"."st_hausdorffdistance"("geom1" "public"."geometry", "geom2" "public"."geometry", float8)
  RETURNS "pg_catalog"."float8" AS '$libdir/postgis-3', 'hausdorffdistancedensify'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_hausdorffdistance"("geom1" "public"."geometry", "geom2" "public"."geometry", float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_hausdorffdistance
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_hausdorffdistance"("geom1" "public"."geometry", "geom2" "public"."geometry");
CREATE FUNCTION "public"."st_hausdorffdistance"("geom1" "public"."geometry", "geom2" "public"."geometry")
  RETURNS "pg_catalog"."float8" AS '$libdir/postgis-3', 'hausdorffdistance'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_hausdorffdistance"("geom1" "public"."geometry", "geom2" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_hexagon
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_hexagon"("size" float8, "cell_i" int4, "cell_j" int4, "origin" "public"."geometry");
CREATE FUNCTION "public"."st_hexagon"("size" float8, "cell_i" int4, "cell_j" int4, "origin" "public"."geometry"='010100000000000000000000000000000000000000'::geometry)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'ST_Hexagon'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_hexagon"("size" float8, "cell_i" int4, "cell_j" int4, "origin" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_hexagongrid
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_hexagongrid"("size" float8, "bounds" "public"."geometry", OUT "geom" "public"."geometry", OUT "i" int4, OUT "j" int4);
CREATE FUNCTION "public"."st_hexagongrid"(IN "size" float8, IN "bounds" "public"."geometry", OUT "geom" "public"."geometry", OUT "i" int4, OUT "j" int4)
  RETURNS SETOF "pg_catalog"."record" AS '$libdir/postgis-3', 'ST_ShapeGrid'
  LANGUAGE c IMMUTABLE STRICT
  COST 250
  ROWS 1000;
ALTER FUNCTION "public"."st_hexagongrid"("size" float8, "bounds" "public"."geometry", OUT "geom" "public"."geometry", OUT "i" int4, OUT "j" int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_interiorringn
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_interiorringn"("public"."geometry", int4);
CREATE FUNCTION "public"."st_interiorringn"("public"."geometry", int4)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'LWGEOM_interiorringn_polygon'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_interiorringn"("public"."geometry", int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_interpolatepoint
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_interpolatepoint"("line" "public"."geometry", "point" "public"."geometry");
CREATE FUNCTION "public"."st_interpolatepoint"("line" "public"."geometry", "point" "public"."geometry")
  RETURNS "pg_catalog"."float8" AS '$libdir/postgis-3', 'ST_InterpolatePoint'
  LANGUAGE c IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_interpolatepoint"("line" "public"."geometry", "point" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_intersection
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_intersection"("geom1" "public"."geometry", "geom2" "public"."geometry", "gridsize" float8);
CREATE FUNCTION "public"."st_intersection"("geom1" "public"."geometry", "geom2" "public"."geometry", "gridsize" float8='-1'::integer)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'ST_Intersection'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_intersection"("geom1" "public"."geometry", "geom2" "public"."geometry", "gridsize" float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_intersection
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_intersection"(text, text);
CREATE FUNCTION "public"."st_intersection"(text, text)
  RETURNS "public"."geometry" AS $BODY$ SELECT public.ST_Intersection($1::public.geometry, $2::public.geometry);  $BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_intersection"(text, text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_intersection
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_intersection"("public"."geography", "public"."geography");
CREATE FUNCTION "public"."st_intersection"("public"."geography", "public"."geography")
  RETURNS "public"."geography" AS $BODY$SELECT public.geography(public.ST_Transform(public.ST_Intersection(public.ST_Transform(public.geometry($1), public._ST_BestSRID($1, $2)), public.ST_Transform(public.geometry($2), public._ST_BestSRID($1, $2))), public.ST_SRID($1)))$BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 100;
ALTER FUNCTION "public"."st_intersection"("public"."geography", "public"."geography") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_intersects
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_intersects"(text, text);
CREATE FUNCTION "public"."st_intersects"(text, text)
  RETURNS "pg_catalog"."bool" AS $BODY$ SELECT public.ST_Intersects($1::public.geometry, $2::public.geometry);  $BODY$
  LANGUAGE sql IMMUTABLE
  COST 100;
ALTER FUNCTION "public"."st_intersects"(text, text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_intersects
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_intersects"("geom1" "public"."geometry", "geom2" "public"."geometry");
CREATE FUNCTION "public"."st_intersects"("geom1" "public"."geometry", "geom2" "public"."geometry")
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'ST_Intersects'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_intersects"("geom1" "public"."geometry", "geom2" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_intersects
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_intersects"("geog1" "public"."geography", "geog2" "public"."geography");
CREATE FUNCTION "public"."st_intersects"("geog1" "public"."geography", "geog2" "public"."geography")
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'geography_intersects'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_intersects"("geog1" "public"."geography", "geog2" "public"."geography") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_inversetransformpipeline
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_inversetransformpipeline"("geom" "public"."geometry", "pipeline" text, "to_srid" int4);
CREATE FUNCTION "public"."st_inversetransformpipeline"("geom" "public"."geometry", "pipeline" text, "to_srid" int4=0)
  RETURNS "public"."geometry" AS $BODY$SELECT public.postgis_transform_pipeline_geometry($1, $2, FALSE, $3)$BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_inversetransformpipeline"("geom" "public"."geometry", "pipeline" text, "to_srid" int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_isclosed
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_isclosed"("public"."geometry");
CREATE FUNCTION "public"."st_isclosed"("public"."geometry")
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'LWGEOM_isclosed'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_isclosed"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_iscollection
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_iscollection"("public"."geometry");
CREATE FUNCTION "public"."st_iscollection"("public"."geometry")
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'ST_IsCollection'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."st_iscollection"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_isempty
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_isempty"("public"."geometry");
CREATE FUNCTION "public"."st_isempty"("public"."geometry")
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'LWGEOM_isempty'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_isempty"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_ispolygonccw
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_ispolygonccw"("public"."geometry");
CREATE FUNCTION "public"."st_ispolygonccw"("public"."geometry")
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'ST_IsPolygonCCW'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_ispolygonccw"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_ispolygoncw
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_ispolygoncw"("public"."geometry");
CREATE FUNCTION "public"."st_ispolygoncw"("public"."geometry")
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'ST_IsPolygonCW'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_ispolygoncw"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_isring
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_isring"("public"."geometry");
CREATE FUNCTION "public"."st_isring"("public"."geometry")
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'isring'
  LANGUAGE c IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_isring"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_issimple
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_issimple"("public"."geometry");
CREATE FUNCTION "public"."st_issimple"("public"."geometry")
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'issimple'
  LANGUAGE c IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_issimple"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_isvalid
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_isvalid"("public"."geometry");
CREATE FUNCTION "public"."st_isvalid"("public"."geometry")
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'isvalid'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_isvalid"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_isvalid
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_isvalid"("public"."geometry", int4);
CREATE FUNCTION "public"."st_isvalid"("public"."geometry", int4)
  RETURNS "pg_catalog"."bool" AS $BODY$SELECT (public.ST_isValidDetail($1, $2)).valid$BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_isvalid"("public"."geometry", int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_isvaliddetail
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_isvaliddetail"("geom" "public"."geometry", "flags" int4);
CREATE FUNCTION "public"."st_isvaliddetail"("geom" "public"."geometry", "flags" int4=0)
  RETURNS "public"."valid_detail" AS '$libdir/postgis-3', 'isvaliddetail'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_isvaliddetail"("geom" "public"."geometry", "flags" int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_isvalidreason
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_isvalidreason"("public"."geometry", int4);
CREATE FUNCTION "public"."st_isvalidreason"("public"."geometry", int4)
  RETURNS "pg_catalog"."text" AS $BODY$
	SELECT CASE WHEN valid THEN 'Valid Geometry' ELSE reason END FROM (
		SELECT (public.ST_isValidDetail($1, $2)).*
	) foo
	$BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_isvalidreason"("public"."geometry", int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_isvalidreason
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_isvalidreason"("public"."geometry");
CREATE FUNCTION "public"."st_isvalidreason"("public"."geometry")
  RETURNS "pg_catalog"."text" AS '$libdir/postgis-3', 'isvalidreason'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_isvalidreason"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_isvalidtrajectory
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_isvalidtrajectory"("public"."geometry");
CREATE FUNCTION "public"."st_isvalidtrajectory"("public"."geometry")
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'ST_IsValidTrajectory'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_isvalidtrajectory"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_largestemptycircle
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_largestemptycircle"("geom" "public"."geometry", "tolerance" float8, "boundary" "public"."geometry", OUT "center" "public"."geometry", OUT "nearest" "public"."geometry", OUT "radius" float8);
CREATE FUNCTION "public"."st_largestemptycircle"(IN "geom" "public"."geometry", IN "tolerance" float8=0.0, IN "boundary" "public"."geometry"='0101000000000000000000F87F000000000000F87F'::geometry, OUT "center" "public"."geometry", OUT "nearest" "public"."geometry", OUT "radius" float8)
  RETURNS "pg_catalog"."record" AS '$libdir/postgis-3', 'ST_LargestEmptyCircle'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_largestemptycircle"("geom" "public"."geometry", "tolerance" float8, "boundary" "public"."geometry", OUT "center" "public"."geometry", OUT "nearest" "public"."geometry", OUT "radius" float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_length
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_length"("public"."geometry");
CREATE FUNCTION "public"."st_length"("public"."geometry")
  RETURNS "pg_catalog"."float8" AS '$libdir/postgis-3', 'LWGEOM_length2d_linestring'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_length"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_length
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_length"("geog" "public"."geography", "use_spheroid" bool);
CREATE FUNCTION "public"."st_length"("geog" "public"."geography", "use_spheroid" bool=true)
  RETURNS "pg_catalog"."float8" AS '$libdir/postgis-3', 'geography_length'
  LANGUAGE c IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_length"("geog" "public"."geography", "use_spheroid" bool) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_length
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_length"(text);
CREATE FUNCTION "public"."st_length"(text)
  RETURNS "pg_catalog"."float8" AS $BODY$ SELECT public.ST_Length($1::public.geometry);  $BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 100;
ALTER FUNCTION "public"."st_length"(text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_length2d
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_length2d"("public"."geometry");
CREATE FUNCTION "public"."st_length2d"("public"."geometry")
  RETURNS "pg_catalog"."float8" AS '$libdir/postgis-3', 'LWGEOM_length2d_linestring'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_length2d"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_length2dspheroid
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_length2dspheroid"("public"."geometry", "public"."spheroid");
CREATE FUNCTION "public"."st_length2dspheroid"("public"."geometry", "public"."spheroid")
  RETURNS "pg_catalog"."float8" AS '$libdir/postgis-3', 'LWGEOM_length2d_ellipsoid'
  LANGUAGE c IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_length2dspheroid"("public"."geometry", "public"."spheroid") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_lengthspheroid
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_lengthspheroid"("public"."geometry", "public"."spheroid");
CREATE FUNCTION "public"."st_lengthspheroid"("public"."geometry", "public"."spheroid")
  RETURNS "pg_catalog"."float8" AS '$libdir/postgis-3', 'LWGEOM_length_ellipsoid_linestring'
  LANGUAGE c IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_lengthspheroid"("public"."geometry", "public"."spheroid") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_letters
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_letters"("letters" text, "font" json);
CREATE FUNCTION "public"."st_letters"("letters" text, "font" json=NULL::json)
  RETURNS "public"."geometry" AS $BODY$
DECLARE
  letterarray text[];
  letter text;
  geom geometry;
  prevgeom geometry = NULL;
  adjustment float8 = 0.0;
  position float8 = 0.0;
  text_height float8 = 100.0;
  width float8;
  m_width float8;
  spacing float8;
  dist float8;
  wordarr geometry[];
  wordgeom geometry;
  -- geometry has been run through replace(encode(st_astwkb(geom),'base64'), E'\n', '')
  font_default_height float8 = 1000.0;
  font_default json = '{
  "!":"BgACAQhUrgsTFOQCABQAExELiwi5AgAJiggBYQmJCgAOAg4CDAIOBAoEDAYKBgoGCggICAgICAgGCgYKBgoGCgQMBAoECgQMAgoADAIKAAoADAEKAAwBCgMKAQwDCgMKAwoFCAUKBwgHBgcIBwYJBgkECwYJBAsCDQILAg0CDQANAQ0BCwELAwsDCwUJBQkFCQcHBwcHBwcFCQUJBQkFCQMLAwkDCQMLAQkACwEJAAkACwIJAAsCCQQJAgsECQQJBAkGBwYJCAcIBQgHCAUKBQoDDAUKAQwDDgEMAQ4BDg==",
  "&":"BgABAskBygP+BowEAACZAmcAANsCAw0FDwUNBQ0FDQcLBw0HCwcLCQsJCwkLCQkJCwsJCwkLCQ0HCwcNBw8HDQUPBQ8DDwMRAw8DEQERAREBEQERABcAFQIXAhUCEwQVBBMGEwYTBhEIEQgPChEKDwoPDA0MDQwNDgsOCRAJEAkQBxAHEgUSBRQFFAMUAxQBFgEWARgAigEAFAISABICEgQQAhAEEAQQBg4GEAoOCg4MDg4ODgwSDgsMCwoJDAcMBwwFDgUMAw4DDgEOARABDgEQARIBEAASAHgAIAQeBB4GHAgaChoMGA4WDhYQFBISEhISDhQQFAwWDBYKFgoYBhgIGAQYBBgCGgAaABgBGAMYAxYHFgUWCRYJFAsUCxIPEg0SERARDhMOFQwVDBcIGQYbBhsCHQIfAR+dAgAADAAKAQoBCgEIAwgFBgUGBQYHBAUEBwQHAgcCBwIHAAcABwAHAQcBBwMHAwUDBwUFBQUHBQUBBwMJAQkBCQAJAJcBAAUCBQAFAgUEBQIDBAUEAwQDBgMEAQYDBgEGAAgBBgAKSeECAJ8BFi84HUQDQCAAmAKNAQAvExMx",
  "\"":"BgACAQUmwguEAgAAkwSDAgAAlAQBBfACAIACAACTBP8BAACUBA==",
  "''":"BgABAQUmwguEAgAAkwSDAgAAlAQ=",
  "(":"BgABAUOQBNwLDScNKw0rCysLLwsxCTEJMwc1BzcHNwM7AzsDPwE/AEEANwI1AjMEMwIzBjEGLwYvCC0ILQgrCCkKKQonCicMJbkCAAkqCSoHLAksBywFLgcuBS4FMAMwAzADMgEwATQBMgA0ADwCOgI6BDoEOAY4BjYINgg2CjQKMgoyCjIMMAwwDi7AAgA=",
  ")":"BgABAUMQ3Au6AgAOLQwvDC8KMQoxCjEKMwg1CDUGNQY3BDcEOQI5AjkAOwAzATEBMQExAy8DLwMvBS8FLQctBS0HKwktBykJKwkpswIADCYKKAooCioIKggsCC4ILgYwBjAGMgQ0AjQCNAI2ADgAQgFAAz4DPAM8BzgHOAc2CTQJMgsyCzALLg0sDSoNKg==",
  "+":"BgABAQ3IBOwGALcBuAEAANUBtwEAALcB0wEAALgBtwEAANYBuAEAALgB1AEA",
  "/":"BgABAQVCAoIDwAuyAgCFA78LrQIA",
  "4":"BgABAhDkBr4EkgEAEREApwJ/AADxARIR5QIAEhIA9AHdAwAA7ALIA9AG6gIAEREA8QYFqwIAAIIDwwH/AgABxAEA",
  "v":"BgABASDmA5AEPu4CROwBExb6AgAZFdMC0wgUFaECABIU0wLWCBcW+AIAExVE6wEEFQQXBBUEFwQVBBUEFwQVBBUEFwQVBBUEFwQXBBUEFwYA",
  ",":"BgABAWMYpAEADgIOAgwCDgQMBAoGDAYKBgoICAgICAgICAoGCgYKBAoEDAQKBAoCDAIKAgwCCgAKAAwACgEMAQoBCgMMAwoDCgUKBQgFCgUIBwYJCAcGCQYJBAsGCQQLAg0CCwINAg0AAwABAAMAAwADAQMAAwADAAMBBQAFAQcBBwEHAwcBCQMJAQsDCwMLAw0FDQMNBQ8FDwURBxMFEwkTBxcJFwkXswEAIMgBCQYJBgkGBwYJCAcIBQgHCgUKBQoFDAEMAwwBDgEOABA=",
  "-":"BgABAQUq0AMArALEBAAAqwLDBAA=",
  ".":"BgABAWFOrAEADgIOAg4CDgQMBAoGDAYKBgoICAgKCAgIBgoGCgYKBgoEDAQKBAwECgIMAAwCDAAMAAwBCgAMAQoDDAMKAwoDCgUKBQgFCgUIBwgJBgcICQYJBgsGCQQLAg0CDQINAA0ADQENAQ0BCwMNAwkFCwUJBQkHBwcJBwUHBwkFCQUJBQkDCwMJAwsDCQELAAsBCwALAAsCCQALAgkECwQJBAkECQYJBgcGBwgJBgcKBQgHCgUKBQwFCgEOAwwBDgEOAA4=",
  "0":"BgABAoMB+APaCxwAHAEaARoDFgMYBRYFFAcUBxIJEgkQCRALEAsOCwwNDA0MDQoPCg0IDwgPBhEGDwYRBA8EEQIRAhMCEQITABMA4QUAEQETAREBEQMRAxEFEQURBREHDwkPBw8JDwsNCw0LDQ0NDQsNCw8JEQkRCREJEwcTBxUFFQUVAxUDFwEXARkAGQAZAhcCFwQXBBUGEwYTCBMIEQoRCg8KDwoPDA0MDQ4NDgsOCQ4JEAkQBxAHEAUSBRIDEgMSAxIDEgESARQAEgDiBQASAhQCEgISBBIEEgYSBhIGEggQChAIEAoQDBAMDgwODg4ODA4MEgwQChIKEggUCBQIFgYWBBYGGAQYAhgCGgILZIcDHTZBEkMRHTUA4QUeOUITRBIePADiBQ==",
  "2":"BgABAWpUwALUA44GAAoBCAEKAQgDBgMGBQYFBgUEBwQFBAUCBwIHAgUABwAHAAUBBwMFAQcFBQMHBQUHBQcFBwMJAwkBCQELAQsAC68CAAAUAhIAFAISBBQCEgQUBBIEEgYUCBIGEAgSChAKEAoQDBAMDg4ODgwQDBIMEgoSChQIFggWCBgGGAQaAhwCHAIWABQBFgEUARQDFAMSAxQFEgUSBxIHEAkQCRALDgsODQ4NDA8KDwwRCBMKEwgTBhUGFwQXBBcEGwAbABsAHQEftwPJBdIDAACpAhIPzwYAFBIArgI=",
  "1":"BgABARCsBLALAJ0LEhERADcA2QEANwATABQSAOYIpwEAALgCERKEBAASABER",
  "3":"BgABAZ0B/gbEC/sB0QQOAwwBDAMMAwwFCgMKBQoFCgUIBwoFCAcICQgJBgkICQYLCAsECwYLBA0GDwINBA8CDwQRAhECEQITABUCFQAVAH0AEQETAREBEQETAxEDEQURBREFDwcRBw8JDwkNCQ8LDQsNDQsNCw0LDwsPCREJEQcRBxMFFQUVBRUDFwEXARkAGQAZAhkCFwQVBBUEEwYTCBEIEQgRCg0MDwoNDA0OCw4LDgkQCRAHEAkQBRAFEgUSAxIDFAMSAxYBFAEWARYAFqQCAAALAgkCCQQHAgcGBwYHBgUIBQYDCAMIAwYDCAEIAQgACAAIAAgCCAIIAgYCCAQIBAgGBgYEBgQIBAoCCgAKAAwAvAEABgEIAAYBBgMGAwQDBgMEBQQDBAUCBQQFAgUABwIFAJkBAACmAaIB3ALbAgAREQDmAhIRggYA",
  "5":"BgABAaAB0APgBxIAFAESABIBEgMSARADEgMQAxIFEAcOBRAHDgkOCQ4JDgsMCwwLCgsKDQoPCA0IDwgPBhEEEwYTAhMEFwIXABcAiQIAEwETABEBEQMTAxEDDwMRBQ8FDwUPBw8JDQcNCQ0LDQsLCwsNCw0JDwkPCREHEQcTBxMFEwMVAxcDGQEZARkAFwAVAhUCFQQTBBMGEwYRCBEIDwoPCg8KDQwNDA0MCw4LDgkOCRAJEAcOBxAHEgUQBRIDEAMSAxIBEgEUARIAFLgCAAAFAgUABQIFBAUCBQQDBAUEAwYDBgMIAwgBCAEIAQoACAAIAgYACAQGAgQEBgQEBAQGBAQCBgIGAgYCBgIIAAYA4AEABgEIAAYBBgMGAQQDBgMEAwQFBAMCBQQFAgUABwIFAPkBAG+OAQCCBRESAgAAAuYFABMRAK8CjQMAAJ8BNgA=",
  "7":"BgABAQrQBsILhQOvCxQR7wIAEhK+AvYIiwMAAKgCERKwBgA=",
  "6":"BgABAsYBnAOqBxgGFgYYBBYEFgIWABQBFgEUAxQDFAUUBRIFEAcSCRAJEAkOCw4NDgsMDQoPCg8KDwgRCBEGEQYRBBMCEwITAhUAkwIBAAERAREBEQEPAxEFEQMPBREFDwcPBw8HDwkNCQ0LDQsNCwsNCw0LDQkPCQ8JDwcRBxEHEwUTAxMFFQEXAxcBGQAVABUCEwIVBBMEEQYTBhEIEQgPChEKDQoPDA0MDQwNDgsOCxALDgkQCRAHEgcQBxIFEgUSBRIBFAMSARIBFAASAOIFABACEgIQAhIEEAQQBhIGEAYQCBAKEAgOChAMDgwMDA4ODA4MDgwODBAKEAoQChIIEggSBhQGFgYUAhYCGAIYABoAGAEYARYBFgMUBRQFEgUSBxAHEAcQCQ4LDgkMCwwNDA0KDQgPCg0GEQgPBhEEEQQRBBMEEwITAhMCFQIVABWrAgAACgEIAQoBCAEGAwYDBgUGBQQFBAUEBQQFAgUABwIFAAUABwEFAAUBBQMFAwUDBQMFBQMFAwUBBQEHAQkBBwAJAJcBDUbpBDASFi4A4AETLC8SBQAvERUrAN8BFC0yEQQA",
  "8":"BgABA9gB6gPYCxYAFAEUARYBEgMUBRQFEgUSBxIHEAcSCQ4JEAkOCw4LDgsMDQwNCg0KDQoPCg8IDwgPBhEGEQQPBBMCEQIRABMAQwAxAA8BEQEPAREDDwMRAw8FEQUPBxEJDwkPCQ8NDw0PDQ8IBwYHCAcGBwgHBgkGBwYJBgcECQYJBAkGCQQJBAsECwQLBA0CCwINAg8CDwIPAA8AaQATAREBEwERAxEFEQURBREHEQcPBw8JDwkPCw8LDQsNDQ0LCw0LDwsNCQ8JDwcPBw8HEQURAxEFEQMRARMBEwFDABEAEwIRAhEEEQQRBg8GEQgPCA8KDwoPCg0MDQwNDAsOCw4LDgkQCRAJDgkQBxIHEAcSBRADEgMUAxIBFAEUABQAagAOAhAADgIOAg4EDAIOBAwEDAQMBgwECgYMBAoGCAYKBgoGCggKBgoICgYICAoICA0MCwwLDgsOCRAHEAcQBxIFEgUSAxIDEgMSARABEgASADIARAASAhICEgQSAhIGEAYSBhAIEAgQCBAKDgoODA4MDgwMDgwODA4KEAwQCBIKEggSCBQIFAYUBBQEFgQWAhYCGAANT78EFis0EwYANBIYLgC0ARcsMRQFADERGS0AswELogHtAhcuNxA3DRkvALMBGjE6ETYSGDIAtAE=",
  "9":"BgABAsYBpASeBBcFFQUXAxUDFQEVABMCFQITBBMEEwYRBhMGDwgRCg8KDwoNDA0OCwwNDgkQCRAJEAcSBxIFEgUSAxQBFAEUARYAlAICAAISAhICEgQSAhAGEgQQBhIGEAgSCA4IEAoOChAMDAwODAwODA4MEAoOChAKEAgSCBIIFAYUBBQGFgIYBBgCGgAWABYBFAEWAxQDEgUUBRIHEgcQCRIJEAkOCw4LDgsODQwNDA0MDwoPCg8IDwgRCBEGEQYRBhEEEQITAhECEwARAOEFAA8BEQEPAREDDwMPBREFDwUPBw8JDwcNCQ8LDQsLCw0NCw0LDQsNCw8JEQkPCREHEQcTBRMFEwUTARUBFQEXABkAFwIXAhcCFQQTBhMGEQYRCA8IDwgNCg8MCwoLDAsOCQ4JDgkQBxAHEAUQBRIFEgMSAxQDFAEUAxQAFgEWABamAgAACwIJAgkCCQIHBAcEBwYFBgUGAwYDBgMGAQgBBgEIAAgABgIIAgYCBgQGBAYEBgYGBgQIBAgECAIKAgoCCgAMAJgBDUXqBC8RFS0A3wEUKzARBgAwEhYsAOABEy4xEgMA",
  ":":"BgACAWE0rAEADgIOAg4CDgQMBAoGDAYKBgoICAgKCAgIBgoGCgYKBgoEDAQKBAwECgIMAAwCDAAMAAwBCgAMAQoDDAMKAwoDCgUKBQgFCgUIBwgJBgcICQYJBgsGCQQLAg0CDQINAA0ADQENAQ0BCwMNAwkFCwUJBQkHBwcJBwUHBwkFCQUJBQkDCwMJAwsDCQELAAsBCwALAAsCCQALAgkECwQJBAkECQYJBgcGBwgJBgcKBQgHCgUKBQwFCgEOAwwBDgEOAA4BYQDqBAAOAg4CDgIOBAwECgYMBgoGCggICAoICAgGCgYKBgoGCgQMBAoEDAQKAgwADAIMAAwADAEKAAwBCgMMAwoDCgMKBQoFCAUKBQgHCAkGBwgJBgkGCwYJBAsCDQINAg0ADQANAQ0BDQELAw0DCQULBQkFCQcHBwkHBQcHCQUJBQkFCQMLAwkDCwEJAwsACwELAAsACwIJAAsECQILBAkECQQJBgkGBwYHCAkGBwoFCAcKBQoFDAUKAQ4DDAEOAQ4ADg==",
  "x":"BgABARHmAoAJMIMBNLUBNrYBMIQB1AIA9QG/BI4CvwTVAgA5hgFBwAFFxwE1fdUCAI4CwATzAcAE1AIA",
  ";":"BgACAWEslgYADgIOAg4CDgQMBAoGDAYKBgoICAgKCAgIBgoGCgYKBgoEDAQKBAwECgIMAAwCDAAMAAwBCgAMAQoDDAMKAwoDCgUKBQgFCgUIBwgJBgcICQYJBgsGCQQLAg0CDQINAA0ADQENAQ0BCwMNAwkFCwUJBQkHBwcJBwUHBwkFCQUJBQkDCwMJAwsBCQMLAAsBCwALAAsCCQALBAkCCwQJBAkECQYJBgcGBwgJBgcKBQgHCgUKBQwFCgEOAwwBDgEOAA4BYwjxBAAOAg4CDAIOBAwECgYMBgoGCggICAgICAgICgYKBgoECgQMBAoECgIMAgoCDAIKAAoADAAKAQwBCgEKAwwDCgMKBQoFCAUKBQgHBgkIBwYJBgkECwYJBAsCDQILAg0CDQADAAEAAwADAAMBAwADAAMAAwEFAAUBBwEHAQcDBwEJAwkBCwMLAwsDDQUNAw0FDwUPBREHEwUTCRMHFwkXCRezAQAgyAEJBgkGCQYHBgkIBwgFCAcKBQoFCgUMAQwDDAEOAQ4AEA==",
  "=":"BgACAQUawAUA5gHEBAAA5QHDBAABBQC5AgDsAcQEAADrAcMEAA==",
  "B":"BgABA2e2BMQLFgAUARQBFAEUAxIDEgUSBRIFEAcQBxAJDgkOCQ4LDgsMCwwNDA0KDQgNCg0IDwYPBg8GDwQRBBEEEQIRAhMAEwAHAAkABwEHAAkBCQAHAQkBCQEHAQkBCQMJAwcDCQMJAwkFBwUJAwkHCQUHBQkHCQcJBwcHBwkHBwcJBwsHCQUQBQ4FDgcOCQ4JDAkMCwoNCg0IDwgRBhMEFQQXAhcCGwDJAQEvAysFJwklDSMPHREbFRkXFRsTHw8fCyUJJwcrAy0B6wMAEhIAoAsREuYDAAiRAYEElgEAKioSSA1EOR6JAQAA0wEJkAGPBSwSEiwAzAETKikSjwEAAMUCkAEA",
  "A":"BgABAg/KBfIBqQIAN98BEhHzAgAWEuwCngsREvwCABMR8gKdCxIR8QIAFBI54AEFlwGCBk3TA6ABAE3UAwMA",
  "?":"BgACAe4BsgaYCAAZABkBFwEXBRUDEwUTBxEHEQcPCQ8JDQkNCQ0LCwsLCwsLCQsJCwcNBwsHDQcLBQsFDQULAwkFCwMLAwkDCQMBAAABAQABAAEBAQABAAEAAQABAAABAQAAAQEAEwcBAQABAAMBAwADAAUABQAFAAcABwAFAAcABwAFAgcABQAHAAUAW7cCAABcABgBFgAUAhQAFAISAhACEAIQBA4EDgQMBgwGDAYMBgoICgYKCAgKCggICAgKBgoICgYMCAwGDAgOBg4GEAYQBgIAAgIEAAICBAACAgQCBAIKBAoGCAQKBggIBgYICAYIBggGCgQIBAoECAQKAggCCgIKAAgACgAKAAgBCAEKAwgDCAMIAwgFBgMIBQYHBAUGBQQFBAcCBQQHAgcCCQIHAgkCBwAJAgkACQAJAAkBCQAJAQsACQELAQsDCwELAwsDCwMLAwsDCwULAwsFCwMLBV2YAgYECAQKBAwGDAQMBhAIEAYSBhIIEgYUBhIEFgYUBBYEFgQWAhgCFgIYABYAGAAYARgBGAMWBRYHFgcWCRYLFA0IBQYDCAUIBwYFCAcGBwgHBgcICQYJCAkGCQYJCAsGCwYLBgsGDQYNBA0GDQQNBA8EDwQPAg8EEQIRAhEAEQITAWGpBesGAA4CDgIOAg4EDAQKBgwGCgYKCAgICggICAYKBgoGCgYKBAwECgQMBAoCDAAMAgwADAAMAQoADAEKAwwDCgMKAwoFCgUIBQoFCAcICQYHCAkGCQYLBgkECwINAg0CDQANAA0BDQENAQsDDQMJBQsFCQUJBwcHCQcFBwcJBQkFCQUJAwsDCQMLAwkBCwALAQsACwALAgkACwIJBAsECQQJBAkGCQYHBgcICQYHCgUIBwoFCgUMBQoBDgMMAQ4BDgAO",
  "C":"BgABAWmmA4ADAAUCBQAFAgUEBQIDBAUEAwQDBgMEAQYDBgEGAAgBBgDWAgAAwQLVAgATABMCEQITBBEEEQQRBhEIEQgPCA8KDwoNCg0MDQwNDAsOCw4LDgkOCxAHEAkQBxIHEgUSBRIDEgEUARIBFAAUAMIFABQCFAISBBQEEgQSBhIIEggSCBAKEAoQCg4MDgwODA4ODA4MDgwQDA4KEggQChIIEggSBhIGFAQSAhQCEgIUAMYCAADBAsUCAAUABwEFAAUBBQMDAQUDAwMDAwMFAQMDBQEFAAUBBwAFAMEF",
  "L":"BgABAQmcBhISEdkFABIQALQLwgIAAIEJ9AIAAK8C",
  "D":"BgABAkeyBMQLFAAUARIBFAESAxIDEgMSBRIFEAcQBxAHDgkOCQ4LDgsMCwwNDA0KDwoPCg8IDwgRCBEGEwQTBBMEEwIVAhUAFwDBBQAXARcBFwMTAxUDEwUTBxEHEQcPCQ8JDwkNCw0LCwsLDQsNCQ0JDQcPBw8HDwcRBREFEQMRAxEDEwERARMBEwDfAwASEgCgCxES4AMACT6BAxEuKxKLAQAAvwaMAQAsEhIsAMIF",
  "F":"BgABARGABoIJ2QIAAIECsgIAEhIA4QIRErECAACvBBIR5QIAEhIAsgucBQASEgDlAhES",
  "E":"BgABARRkxAuWBQAQEgDlAhES0QIAAP0BtgIAEhIA5wIRFLUCAAD/AfACABISAOUCERLDBQASEgCyCw==",
  "G":"BgABAZsBjgeIAgMNBQ8FDQUNBQ0HCwcNBwsHCwkLCQsJCwsJCwsLCQsJDQkLBw0HDwcNBw8FDwUPAw8DEQMPAxEBEQERARMBEQAXABUCFwIVAhMEFQQTBhMGEwYRCBEIDwoRCg8KDwwNDA0MDQ4LDgkQCRAJEAcQBxIFEgUUBRQDFAMUARYBFgEYAMoFABQCFAASBBQCEgQSBBIEEgYSBhAGEAgQCBAKDgoOCg4MDgwMDgwOChAKEAoSCBIIFAgUBhQEGAYWAhgEGAIaAOoCAAC3AukCAAcABwEFAQUBBQMFAwMFAwUDBQEFAQcBBQEFAQUABwAFAMUFAAUCBwIFAgUCBQQFBAMGBQYDBgUGAwgDBgMIAQgDCAEIAQoBCAEIAAgACgAIAAgCCAIIAggECgQGBAgECAYIBgC6AnEAAJwCmAMAAJcF",
  "H":"BgABARbSB7ILAQAAnwsSEeUCABISAOAE5QEAAN8EEhHlAgASEgCiCxEQ5gIAEREA/QPmAQAAgAQPEOYCABER",
  "I":"BgABAQmuA7ILAJ8LFBHtAgAUEgCgCxMS7gIAExE=",
  "J":"BgABAWuqB7ILALEIABEBEwERAREDEwMRAxEFEQURBw8HEQcPCQ0LDwsNCw0NDQ0LDwsPCxEJEQkTCRMJFQcVBxcFFwMZAxsBGwEbAB8AHQIbAhsEGQYXBhcGFQgTCBMKEwoRDA8KDwwNDA0OCw4LDgkQCRAJEAcQBRIFEgUSAxQDEgESARIBFAESABIAgAEREtoCABERAn8ACQIHBAcEBwYHBgUIBQoDCgMKAwoDDAEKAQwBCgEMAAwACgAMAgoCDAIKBAoECgYKBggGBgYGCAQGBAgCCgAIALIIERLmAgAREQ==",
  "M":"BgACAQRm1gsUABMAAAABE5wIAQDBCxIR5QIAEhIA6gIK5gLVAe0B1wHuAQztAgDhAhIR5QIAEhIAxAsUAPoDtwT4A7YEFgA=",
  "K":"BgABAVXMCRoLBQsDCQMLAwsDCwMLAwsBCwELAQsBCwELAQ0ACwELAAsADQALAg0ACwILAA0CCwILAgsCDQQLBAsECwYNBAsGCwYLCAsGCwgJCgsICQoJCgkMCQwJDAkOCRALEAkQCRKZAdICUQAAiwQSEecCABQSAKALExLoAgAREQC3BEIA+AG4BAEAERKCAwAREdkCzQXGAYUDCA0KDQgJCgkMBwoFDAUMAQwBDgAMAg4CDAQOBAwGDghmlQI=",
  "O":"BgABAoMBsATaCxwAHAEaARoDGgMYBRYFFgcWBxQJEgkSCRILEAsODQ4NDg0MDwoNDA8KDwgPCBEIDwYRBg8GEQQRAhMCEQITABMA0QUAEQETAREBEQMTBREFEQURBxEHDwcRCQ8LDQsPCw0NDQ0NDwsPCw8LEQkTCRMJEwkVBxUHFwUXAxkDGQEbARsAGwAZAhkCGQQXBhcGFQYVCBUIEwoRChEMEQoRDA8MDQ4NDg0OCxAJEAsQCRAHEgcSBxIFFAMSAxIDEgEUARIAEgDSBQASAhQCEgISBBIEEgYSBhIIEggQCBAKEgwODBAMEA4ODg4QDhIMEAwSChQKFAgUCBYIFgYYBBoGGgQcAh4CHgILggGLAylCWxZbFSlBANEFKklcGVwYKkwA0gU=",
  "N":"BgABAQ+YA/oEAOUEEhHVAgASEgC+CxQAwATnBQDIBRMS2AIAExEAzQsRAL8ElgU=",
  "P":"BgABAkqoB5AGABcBFQEVAxMDEwMTBREHEQcRBw8JDwkNCQ0LDQsNCwsNCw0JDQkNCQ8HDwcPBxEFEQURAxEDEQMTAREBEwETAH8AAIMDEhHlAgASEgCgCxES1AMAFAAUARIAFAESAxIDEgMSAxIFEAUQBRAHDgkOCQ4JDgsMCwwNDA0KDQoNCg8IDwgRCBEGEwQTBBUEFQIXAhkAGQCzAgnBAsoCESwrEn8AANUDgAEALBISLgDYAg==",
  "R":"BgABAj9msgsREvYDABQAFAESARQBEgESAxIDEgUSBRAFEAcQBw4JDgkOCQ4LDAsMDQwLCg0KDwoNCA8IDwgPBhEEEwYTAhMEFQIXABcAowIAEwEVARMDEwMTBRMFEQcTBxELEQsRDQ8PDREPEQ0VC8QB/QMSEfkCABQSiQGyA3EAALEDFBHnAgASEgCgCwnCAscFogEALhISLACqAhEsLRKhAQAApQM=",
  "Q":"BgABA4YBvAniAbkB8wGZAYABBQUFAwUFBQUHBQUDBwUFBQcFBQMHBQcDBwUJAwcDCQMJAwkDCQMJAQsDCwMLAQsDCwENAw0BDQEPAA8BDwAPABsAGwIZAhcEGQQXBBUGFQgVCBMIEQoTChEKDwwPDA8ODQ4NDgsQCxAJEAkQBxIHEgUSBRQFFAMUARQDFAEWABYAxgUAEgIUAhICEgQSBBIGEgYSCBIIEAgQChIMDgwQDBAODg4OEA4SDBAMEgoUChQIFAgWCBYGGAQaBhoEHAIeAh4CHAAcARoBGgMaAxgFFgUWBxYHFAkSCRIJEgsQCw4NDg0ODQwPCg0MDwoPCA8IEQgPBhEGDwYRBBECEwIRAhMAEwC7BdgBrwEImQSyAwC6AylAWxZbFSk/AP0BjAK7AQeLAoMCGEc4J0wHVBbvAaYBAEM=",
  "S":"BgABAYMC8gOEBxIFEgUQBxIFEgcSBxIJEgcSCRIJEAkQCRALEAsOCw4NDg0MDQ4PDA0KEQoPChEKEQgRCBMGFQQTBBcCFQAXABkBEwARAREBEQMPAQ8DDwMPAw0DDQUNAw0FCwULBwsFCwUJBwsFCQcHBQkHCQUHBwcHBwUHBwUFBQcHBwUHAwcFEQsRCxMJEwkTBxMFEwUVBRUDFQMVARMBFwEVABUAFQIVAhUCFQQVBBUEEwYVBhMIEwgTCBMIEwgRCBMKEQgRCmK6AgwFDgUMAw4FEAUOBRAFEAUQBRAFEAMSAw4DEAMQAxABEAEOAQ4AEAIMAg4CDgQMBAwGCggKCAoKBgwGDgYQBBACCgAMAAoBCAMKBQgFCAcIBwgJCAsGCQgLCA0IDQgNCA8IDQgPCA8IDwgPChEIDwgPCBEKDwoPDBEMDwwPDg8ODw4NEA0QCxALEgsSCRIHEgcUBRQFGAUYAxgBGgEcAR4CJAYkBiAIIAweDBwQHBAYEhgUFBYUFhQWEBoQGg4aDBwKHAoeBh4GIAQgAiACIgEiASIFIgUiBSAJIgkgCyINZ58CBwQJAgkECwQLAgsECwINBA0CDQQNAg0CDQALAg0ADQANAAsBCwELAQsDCwULBQkFCQcHBwcJBwkFCwMLAw0BDQENAAsCCwQLBAkGCQgJCAkKBwoJCgcMBQoHDAcMBQwF",
  "V":"BgABARG2BM4DXrYEbKwDERL0AgAVEesCnQsSEfsCABQS8QKeCxES8gIAExFuqwNgtQQEAA==",
  "T":"BgABAQskxAv0BgAAtQKVAgAA+wgSEeUCABISAPwImwIAALYC",
  "U":"BgABAW76B7ALAKMIABcBFwMXARUFFQUTBxMHEwkRCREJEQsPDQ0LDw0NDwsPCw8LEQkPCRMJEQcTBxMFEwUVBRUDEwMXARUBFQEXABUAEwIVAhMCFQQTBBUEEwYTBhMIEwgRChEIEQwRDA8MDw4PDg0OCxANEAsSCRIJEgcUBxQHFAMWBRYBGAEYARgApggBAREU9AIAExMAAgClCAALAgkECQQHBAcIBwgHCAUKBQoDCgMKAwwBCgEMAQwADAAMAgoCDAIKAgoECgQKBggGCAYICAYKBAgCCgIMAgwApggAARMU9AIAExM=",
  "X":"BgABARmsCBISEYkDABQSS54BWYICXYkCRZUBEhGJAwAUEtYCzgXVAtIFExKIAwATEVClAVj3AVb0AVKqAREShgMAERHXAtEF2ALNBQ==",
  "W":"BgABARuODcQLERHpAp8LFBHlAgASEnW8A2+7AxIR6wIAFBKNA6ALERKSAwATEdQB7wZigARZ8AIREugCAA8RaKsDYsMDXsoDaqYDExLqAgA=",
  "Y":"BgABARK4BcQLhgMAERHnAvMGAKsEEhHnAgAUEgCsBOkC9AYREoYDABERWOEBUJsCUqICVtwBERI=",
  "Z":"BgABAQmAB8QLnwOBCaADAADBAusGAMgDggmhAwAAwgLGBgA=",
  "`":"BgABAQfqAd4JkQHmAQAOlgJCiAGpAgALiwIA",
  "c":"BgABAW3UA84GBQAFAQUABQEFAwMBBQMDAwMDAwUBAwMFAQUABQEHAAUAnQMABQIFAAUCBQQFAgMEBQQDBAMGAwQBBgMGAQYABgEGAPABABoMAMsCGw7tAQATABMCEwARAhMEEQIPBBEEDwQPBg8IDwYNCA0KDQoNCgsMCwwLDAkOCRAHDgcQBxIFEgUUBRQDFAEWAxgBGAAYAKQDABQCFAISBBQCEgYSBhAGEggQCBAIEAoQCg4MDAwODAwODAwKDgwQCg4IEAgQCBAIEAYSBhIGEgQSAhQCFAIUAOABABwOAM0CGQzbAQA=",
  "a":"BgABApoB8AYCxwF+BwkHCQcJCQkHBwkHBwcJBQkFBwUJBQkFCQMHBQkDCQMJAwcDCQEHAQkBBwEJAQcABwAHAQcABQAHAAUBBQAFABMAEwITAhEEEwQPBBEGDwgPCA0IDwoLCg0KCwwLDAsMCQ4JDgkOBw4HEAcQBRAFEAUSAxADEgESAxIBFAESABQAFAISAhQCEgQSBBIEEgYSBhIIEAgQChAIDgwODA4MDg4MDgwODBAMEAoSCBIKEggUCBQGFgYWBBgEGAIaAhoAcgAADgEMAQoBCgEIAwgDBgUEBQQFBAcCBwIHAgkCCQAJAKsCABcPAMwCHAvCAgAUABYBEgAUARIDFAMQAxIDEAUSBQ4FEAcOCRAJDAkOCwwLDA0MCwoNCg8IDwgPCA8GEQYRBhMEEwIXAhUCFwAZAIMGFwAKmQLqA38ATxchQwgnGiMwD1AMUDYAdg==",
  "b":"BgABAkqmBIIJGAAYARYBFgEUAxQDEgUSBRIFEAcQCQ4HDgkOCw4LDAsMDQoNCg0KDQgPBg8GDwYRBBEEEQQTBBECEwIVAhMAFQD/AgAZARcBFwEXAxUDEwUTBREFEQcPBw8JDwkNCQ0LDQsLCwsNCQ0JDQcPBw8HDwURAxEDEQMTAxMBEwMVARUAFQHPAwAUEgCWCxEY5gIAERkAowKCAQAJOvECESwrEn8AAJsEgAEALBISLgCeAw==",
  "d":"BgABAkryBgDLAXAREQ8NEQ0PDREJDwkRBw8FDwURAw8DDwERAw8BEQEPACMCHwQfCB0MGw4bEhcUFxgVGhEeDSANJAkmBSgDKgEuAIADABYCFAIUAhQCFAQUBBIGEgYSBhAIEAgQCBAKDgoODAwMDAwMDgoOCg4KEAgQCBIGEgYSBhQEFgQWBBYCGAIYAHwAAKQCERrmAgARFwCnCxcADOsCugJGMgDmA3sAKxERLQCfAwolHBUmBSQKBAA=",
  "e":"BgABAqMBigP+AgAJAgkCCQQHBAcGBwYFCAUIBQgDCgMIAQoDCAEKAQoACgAKAAoCCAIKAggECgQIBAgGCAYGBgQIBAoECAIKAAyiAgAAGQEXARcBFwMVBRMFEwURBxEHDwcPCQ8LDQkNCwsNCw0LDQkNBw8JDwcPBQ8FEQURAxEDEwMTAxMBFQAVARcALwIrBCkIJwwlDiESHxQbGBkaFR4TIA0iCyQJKAMqASwAggMAFAIUABIEFAISBBIEEgQSBhIGEAgQCBAIEAoODA4MDgwODgwQDBAKEAoSChIIFAgUCBYGGAQYBhoCGgQcAh4ALgEqAygFJgkkDSANHhEaFRgXFBsSHQ4fDCUIJwQpAi0AGQEXAxcDFQcTBRMJEQkPCw8LDQ0PDQsNDQ8LEQsRCxEJEwkTCRMJEwcTBxUHFQUVBRUHFQUVBRUHFwcVBRUHCs4BkAMfOEUURxEfMwBvbBhAGBwaBiA=",
  "h":"BgABAUHYBJAGAAYBBgAGAQYDBgEEAwYDBAMEBQQDAgUEBQIFAAUCBQB1AAC5BhIT5wIAFhQAlAsRGOYCABEZAKMCeAAYABgBFgEWARQDFAMSBRIFEgUQBxAJDgcOCQ4LDgsMCwwNCg0KDQoNCA8GDwYPBhEEEQQRBBMEEQITAhUCEwAVAO0FFhPnAgAUEgD+BQ==",
  "g":"BgABArkBkAeACQCNCw8ZERkRFxEVExMVERUPFQ8XDRcLGQkZBxsFGwUdAR0BDQALAA0ADQINAAsCDQANAg0CDQILAg0EDQINBA0GDQQNBg0EDQYNCA0GDwgNCA0IDQgPCg0KDwwNDA8MDw4PDqIB7gEQDRALEAkQCQ4JEAcOBw4FDgUOAwwFDgMMAQwBDAEMAQwACgEKAAoACAIIAAgCCAIGAggCBgIGBAYCBgQEAgYEAqIBAQADAAEBAwADAAMABQADAAUAAwAFAAMABQAFAAMABQA3ABMAEwIRAhMCEQQRBBEEEQYRBg8IDwgPCA0KDQoNCg0MCwwLDgsOCQ4JDgkQBxAHEgcSBRIDFAMWAxQBFgEYABgA/gIAFgIWAhQEFgQUBBIGFAgSCBIIEAoSChAKDgwODA4MDg4MDgwODA4KEAgQCBAIEgYSBhIEEgYSBBQCEgIUAhQCOgAQABABDgEQAQ4BEAMOAw4FDgUOBQwFDgcMBQ4HDAkMB4oBUBgACbsCzQYAnAR/AC0RES0AnQMSKy4RgAEA",
  "f":"BgABAUH8A6QJBwAHAAUABwEFAQcBBQEFAwUDBQMDAwMDAwUDAwMFAQUAwQHCAQAWEgDZAhUUwQEAAOMEFhftAgAWFADKCQoSChIKEAoQCg4KDgwOCgwMDAoKDAwMCgwIDAgMCAwIDAYOCAwEDgYMBA4GDAIOBA4CDgQOAg4CDgAOAg4ADgC2AQAcDgDRAhkQowEA",
  "i":"BgACAQlQABISALoIERLqAgAREQC5CBIR6QIAAWELyAoADgIOAgwEDgIKBgwGCgYKCAoGCAgICggIBggGCgYKBAoECgQMBAoCDAIMAgwCDAAMAAwADAEMAQoBDAMKAwoDCgUKBQgFCgUIBwgHCAcICQgJBgkECwQJBA0CCwANAA0ADQELAQ0BCwMJBQsFCQUJBwkFBwcHBwcJBQcFCQUJBQkDCQMLAwkBCwELAQsACwALAAsCCwILAgkCCwIJBAkECQQJBgcGCQYHCAcIBwgHCgUKBQwFCgMMAQwBDgEMAA4=",
  "j":"BgACAWFKyAoADgIOAgwEDgIKBgwGCgYKCAoGCAgICggIBggGCgYKBAoECgQMBAoCDAIMAgwCDAAMAAwADAEMAQoBDAMKAwoDCgUKBQgFCgUIBwgHCAcICQgJBgkECwQJBA0CCwANAA0ADQELAQ0BCwMJBQsFCQUJBwkFBwcHBwcJBQcFCQUJBQkDCQMLAwkBCwELAQsACwALAAsCCwILAgkCCwIJBAkECQQJBgcGCQYHCAcIBwgHCgUKBQwFCgMMAQwBDgEMAA4BO+YCnwwJEQkRCQ8JDwsNCQ0LDQkLCwsJCQsLCQkLBwsHCwcLBwsFCwcNAwsFDQMLBQ0BDQMNAQ0DDQENAQ0ADQENAA0AVwAbDQDSAhoPQgAIAAgABgAIAgYCCAIGAgYEBgQGBAQEBAQEBgQEBAYCBgC4CRES6gIAEREAowo=",
  "k":"BgABARKoA/QFIAC0AYoD5gIAjwK5BJICwwTfAgDDAbIDFwAAnwMSEeUCABISAJILERLmAgAREQCvBQ==",
  "n":"BgABAW1yggmQAU8GBAgEBgQGBgYCCAQGBAYEBgQIAgYECAQGAggEBgIIBAgCCAQIAggCCAIIAgoACAIKAAgCCgAKAgoADAAKAgwAFgAWARQAFAEUAxQDFAMSAxIFEgUQBRIHEAkOBxAJDgsOCwwLDA0MDQoPCA8IEQgRBhEGEwYVBBUEFQIXAhkCGQDtBRQR5QIAFBAA/AUACAEIAQYBCAMGBQQFBgUEBwQFBAcCBwIHAgcCCQIHAAcACQAHAQcABwMHAQUDBwMFAwUFBQUDBQEFAwcBBwAHAPkFEhHjAgASEgDwCBAA",
  "m":"BgABAZoBfoIJigFbDAwMCg4KDggOCA4IDgYQBhAGEAQQBBAEEAISAhACEgAmASQDJAciCyANHhEcFRwXDg4QDBAKEAwQCBAKEggSBhIGEgYSBBQEEgIUAhICFAAUABQBEgEUARIDEgMSAxIFEgUQBxAHEAcQBw4JDgkOCw4LDAsMDQoNCg8KDwgPCBEIEQYRBBMEEwQTAhMCFQAVAP0FEhHlAgASEgCCBgAIAQgBBgEGAwYFBgUEBQQHBAUEBwIHAgcCBwIJAAcABwAJAAcBBwEHAQUBBwMFAwUDBQMDBQMFAwUBBQEHAQcAgQYSEeUCABISAIIGAAgBCAEGAQYDBgUGBQQFBAcEBQQHAgcCBwIHAgkABwAHAAkABwEHAQcBBQEHAwUDBQMFAwMFAwUDBQEFAQcBBwCBBhIR5QIAEhIA8AgYAA==",
  "l":"BgABAQnAAwDrAgASFgDWCxEa6gIAERkA0wsUFw==",
  "y":"BgABAZ8BogeNAg8ZERkRFxEVExMVERUPFQ8XDRcLGQkZBxsFGwUdAR0BDQALAA0ADQINAAsCDQANAg0CDQILAg0EDQINBA0GDQQNBg0EDQYNCA0GDwgNCA0IDQgPCg0KDwwNDA8MDw4PDqIB7gEQDRALEAkQCQ4JEAcOBw4FDgUOAwwFDgMMAQwBDAEMAQwACgEKAAoACAIIAAgCCAIGAggCBgIGBAYCBgQEAgYEAqIBAQADAAEBAwADAAMABQADAAUAAwAFAAMABQAFAAMABQA3ABMAEwIRABECEwQRAg8EEQQPBBEGDwgNCA8IDQgNCg0MDQwLDAkOCw4JDgcQBxAHEgUSBRQFFAMWARgDGAEaABwA9AUTEuQCABEPAP8FAAUCBQAFAgUEBQIDBAUEAwQDBgMEAQYDBgEGAAgBBgCAAQAAvAYREuICABMPAP0K",
  "q":"BgABAmj0A4YJFgAWARQAEgESAxADEAMOAw4FDgUMBQ4HDgcOBwwJDgmeAU4A2QwWGesCABYaAN4DAwADAAMBAwADAAUAAwADAAMABQAFAAUABwAHAQcACQAVABUCFQATAhUCEwQRAhMEEQQRBhEGDwgPCA8IDQoNDA0MCwwLDgkOCRAJEAkQBxIHEgUUBRYDFgMYARoBGgAcAP4CABYCFgIWBBYEFAQSBhQIEggSCBAKEgoQDA4MDgwODg4ODBAMDgwQChIIEAoSCBIGEgYUBhQEFAQWAhYCFgIWAApbkQYSKy4ReAAAjARTEjkRHykJMwDvAg==",
  "p":"BgABAmiCBIYJFgAWARYBFAEWAxQDEgUUBRIFEgcSBxAJEAkQCQ4LDgsOCwwNDA0KDwoPCg8IEQgRCBEGEwQTBhMCFQQVAhUAFQD9AgAbARkBFwMXAxcDEwUTBxMHEQcRCQ8JDQsNCw0LCw0LDQkPCQ0JDwURBxEFEQURAxMDEQMTARUBEwEVARUBFQAJAAcABwAFAAcABQAFAAMAAwADAAUAAwIDAAMAAwIDAADdAxYZ6wIAFhoA2gyeAU0OCgwIDgoMCA4GDgYMBg4GDgQQBBAEEgQUAhQCFgIWAApcoQMJNB8qNxJVEQCLBHgALhISLADwAg==",
  "o":"BgABAoMB8gOICRYAFgEWARQBFgMUAxIDFAUSBRIHEgcQBxAJEAkOCw4LDgsMDQwNCg8KDwoPCg8IEQgRBhMGEwQTBBMCFQIVABcAiwMAFwEVARUDEwMTAxMFEwcRBxEHDwkPCQ8LDQsNCw0NCw0LDwkNCw8HEQkPBxEHEQcRBRMFEwMTAxUDFQEVABUAFQAVAhUCFQITBBMEEwYTBhEGEQgRCA8KDwoPCg0KDQwNDAsOCw4JDgkQCRAJEgcSBxIFFAUUAxQDFgEWARYAFgCMAwAYAhYCFgQUBBQEFAYUCBIIEggQChAKEAwODA4MDg4MDgwQCg4KEgoQChIIEggSBhQGEgYUBBYEFAIWAhYCFgALYv0CHTZBFEMRHTcAjwMcNUITQhIiOACQAw==",
  "r":"BgACAQRigAkQAA8AAAABShAAhAFXDAwODAwKDgoOCBAIDgYQBhAEEAQQBBAEEAISABACEAAQAA4BEAAQARADEAEQAxADEAUSBRIHFAcUCxQLFA0WDVJFsQHzAQsMDQwLCgkICwgLCAkGCQYJBAkGBwIJBAcCBwQHAAcCBwAFAgcABQAHAQUABQEFAQUBBQEDAQUBAwMDAQMDAwEAmwYSEeMCABISAO4IEAA=",
  "u":"BgABAV2KBwGPAVANCQsHDQcNBw0FCwUNBQ0FDQMPAw8DEQMTARMBFQEVABUAFQITABMEEwITBBMEEQQRBhEGDwYRCA8KDQgPCg0MDQwLDAsOCRALDgcQBxIHEgUUBRQFFAMWAxgBGAEYARoA7gUTEuYCABMPAPsFAAcCBwIFBAcCBQYDBgUGAwgDBgMIAQgBCAEIAQoBCAAIAAoACAIIAggCCAIGBAgEBgQGBgYGBAYCBgQIAggACAD6BRES5AIAEREA7wgPAA==",
  "s":"BgABAasC/gLwBQoDCgMMBQ4DDgUOBRAFEAUSBRAHEgcQCRIJEAkSCxALEAsQDRANDg0ODw4PDA8MDwoRChEIEwYTBBcCFQIXABkBGQEXAxcFFQUTBRMHEwcRCREJDwkNCQ8LDQ0LCwsNCw0JDQkPBw8HDwUPBREDEQMRAREDEQETABEBEwARABMADwIRABECEQIRBBMCEwQVBBUEFQYVBhMIFwgVChUKFQxgsAIIAwYDCAMKAQgDCAMKAQoDCgEKAwoBCgMKAQwDCgEKAwoBDAMKAQoBCgEMAQoACgEKAAoBCgAKAQgACgAIAQgABgoECAIKAgoCCgAMAQoBDAUEBwIHBAcEBwIHBAkECQQJBAkECQYLBAkGCwYJBgsGCwYJCAsGCwgJBgsICQgLCAkICwgJCgkKCQoJCgcKCQwHDAcMBwwFDAcMAw4FDAMOAw4BDgMQARAAEAESABIAEgIQAg4CDgIOBA4CDgQMBAwEDAQMBgoECgYKBgoGCgYIBggGCAgIBggGBgYIBgYGBgYGBgYGBAgGBgQIBAYECAQQChIIEggSBhIEEgQSBBQCFAISABQAEgASABIAEgESARIBEAEQAxIDDgMQAxADDgUOBQwDDAMMAwoDCAMIAQYBe6cCAwIDAgUAAwIFAgUCBwIFAgcCBQIHAgUCBwIHAAUCBwIHAgUABwIHAgcABQIHAAcCBwAFAgUABQIFAAUABQIDAAEAAQABAQEAAQEBAQEBAQEBAQEDAQEAAwEBAQMAAwEDAAMBAwADAQMAAwABAQMAAwADAAEAAwIBAAMCAQQDAgE=",
  "t":"BgABAUe8BLACWAAaEADRAhsOaQANAA0ADwINAA0CDQANAg0CDQINBA0CCwYNBA0GCwYNBgsIDQgLCAsKCwgJDAsKCQwJDAkOCQ4HEAcSBxIHEgUUAOAEawAVEQDWAhYTbAAAygIVFOYCABUXAMUCogEAFhQA1QIVEqEBAADzAwIFBAMEBQQDBAMEAwYDBgMGAwYBCAEGAQgBBgEIAAgA",
  "w":"BgABARz8BsAEINYCKNgBERLuAgARD+8B3QgSEc0CABQSW7YCV7UCFBHJAgASEpMC3AgREvACABERmAHxBDDaAVeYAxES7gIAEREo1QE81wIIAA==",
  "z":"BgABAQ6cA9AGuQIAFw8AzAIaC9QFAAAr9wKjBuACABYQAMsCGQyZBgCaA9AG"
   }';
BEGIN

  IF font IS NULL THEN
    font := font_default;
  END IF;

  -- For character spacing, use m as guide size
  geom := ST_GeomFromTWKB(decode(font->>'m', 'base64'));
  m_width := ST_XMax(geom) - ST_XMin(geom);
  spacing := m_width / 12;

  letterarray := regexp_split_to_array(replace(letters, ' ', E'\t'), E'');
  FOREACH letter IN ARRAY letterarray
  LOOP
    geom := ST_GeomFromTWKB(decode(font->>(letter), 'base64'));
    -- Chars are not already zeroed out, so do it now
    geom := ST_Translate(geom, -1 * ST_XMin(geom), 0.0);
    -- unknown characters are treated as spaces
    IF geom IS NULL THEN
      -- spaces are a "quarter m" in width
      width := m_width / 3.5;
    ELSE
      width := (ST_XMax(geom) - ST_XMin(geom));
    END IF;
    geom := ST_Translate(geom, position, 0.0);
    -- Tighten up spacing when characters have a large gap
    -- between them like Yo or To
    adjustment := 0.0;
    IF prevgeom IS NOT NULL AND geom IS NOT NULL THEN
      dist = ST_Distance(prevgeom, geom);
      IF dist > spacing THEN
        adjustment = spacing - dist;
        geom := ST_Translate(geom, adjustment, 0.0);
      END IF;
    END IF;
    prevgeom := geom;
    position := position + width + spacing + adjustment;
    wordarr := array_append(wordarr, geom);
  END LOOP;
  -- apply the start point and scaling options
  wordgeom := ST_CollectionExtract(ST_Collect(wordarr));
  wordgeom := ST_Scale(wordgeom,
                text_height/font_default_height,
                text_height/font_default_height);
  return wordgeom;
END;
$BODY$
  LANGUAGE plpgsql IMMUTABLE
  COST 250
  SET "standard_conforming_strings"="on";
ALTER FUNCTION "public"."st_letters"("letters" text, "font" json) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_linecrossingdirection
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_linecrossingdirection"("line1" "public"."geometry", "line2" "public"."geometry");
CREATE FUNCTION "public"."st_linecrossingdirection"("line1" "public"."geometry", "line2" "public"."geometry")
  RETURNS "pg_catalog"."int4" AS '$libdir/postgis-3', 'ST_LineCrossingDirection'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_linecrossingdirection"("line1" "public"."geometry", "line2" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_lineextend
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_lineextend"("geom" "public"."geometry", "distance_forward" float8, "distance_backward" float8);
CREATE FUNCTION "public"."st_lineextend"("geom" "public"."geometry", "distance_forward" float8, "distance_backward" float8=0.0)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'geometry_line_extend'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_lineextend"("geom" "public"."geometry", "distance_forward" float8, "distance_backward" float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_linefromencodedpolyline
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_linefromencodedpolyline"("txtin" text, "nprecision" int4);
CREATE FUNCTION "public"."st_linefromencodedpolyline"("txtin" text, "nprecision" int4=5)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'line_from_encoded_polyline'
  LANGUAGE c IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_linefromencodedpolyline"("txtin" text, "nprecision" int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_linefrommultipoint
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_linefrommultipoint"("public"."geometry");
CREATE FUNCTION "public"."st_linefrommultipoint"("public"."geometry")
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'LWGEOM_line_from_mpoint'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_linefrommultipoint"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_linefromtext
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_linefromtext"(text, int4);
CREATE FUNCTION "public"."st_linefromtext"(text, int4)
  RETURNS "public"."geometry" AS $BODY$
	SELECT CASE WHEN public.ST_GeometryType(public.ST_GeomFromText($1, $2)) = 'ST_LineString'
	THEN public.ST_GeomFromText($1,$2)
	ELSE NULL END
	$BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_linefromtext"(text, int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_linefromtext
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_linefromtext"(text);
CREATE FUNCTION "public"."st_linefromtext"(text)
  RETURNS "public"."geometry" AS $BODY$
	SELECT CASE WHEN public.ST_GeometryType(public.ST_GeomFromText($1)) = 'ST_LineString'
	THEN public.ST_GeomFromText($1)
	ELSE NULL END
	$BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_linefromtext"(text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_linefromwkb
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_linefromwkb"(bytea);
CREATE FUNCTION "public"."st_linefromwkb"(bytea)
  RETURNS "public"."geometry" AS $BODY$
	SELECT CASE WHEN public.ST_GeometryType(public.ST_GeomFromWKB($1)) = 'ST_LineString'
	THEN public.ST_GeomFromWKB($1)
	ELSE NULL END
	$BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_linefromwkb"(bytea) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_linefromwkb
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_linefromwkb"(bytea, int4);
CREATE FUNCTION "public"."st_linefromwkb"(bytea, int4)
  RETURNS "public"."geometry" AS $BODY$
	SELECT CASE WHEN public.ST_GeometryType(public.ST_GeomFromWKB($1, $2)) = 'ST_LineString'
	THEN public.ST_GeomFromWKB($1, $2)
	ELSE NULL END
	$BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_linefromwkb"(bytea, int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_lineinterpolatepoint
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_lineinterpolatepoint"("public"."geometry", float8);
CREATE FUNCTION "public"."st_lineinterpolatepoint"("public"."geometry", float8)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'LWGEOM_line_interpolate_point'
  LANGUAGE c IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_lineinterpolatepoint"("public"."geometry", float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_lineinterpolatepoint
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_lineinterpolatepoint"(text, float8);
CREATE FUNCTION "public"."st_lineinterpolatepoint"(text, float8)
  RETURNS "public"."geometry" AS $BODY$ SELECT public.ST_LineInterpolatePoint($1::public.geometry, $2);  $BODY$
  LANGUAGE sql IMMUTABLE
  COST 100;
ALTER FUNCTION "public"."st_lineinterpolatepoint"(text, float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_lineinterpolatepoint
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_lineinterpolatepoint"("public"."geography", float8, "use_spheroid" bool);
CREATE FUNCTION "public"."st_lineinterpolatepoint"("public"."geography", float8, "use_spheroid" bool=true)
  RETURNS "public"."geography" AS '$libdir/postgis-3', 'geography_line_interpolate_point'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."st_lineinterpolatepoint"("public"."geography", float8, "use_spheroid" bool) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_lineinterpolatepoints
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_lineinterpolatepoints"(text, float8);
CREATE FUNCTION "public"."st_lineinterpolatepoints"(text, float8)
  RETURNS "public"."geometry" AS $BODY$ SELECT public.ST_LineInterpolatePoints($1::public.geometry, $2);  $BODY$
  LANGUAGE sql IMMUTABLE
  COST 100;
ALTER FUNCTION "public"."st_lineinterpolatepoints"(text, float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_lineinterpolatepoints
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_lineinterpolatepoints"("public"."geography", float8, "use_spheroid" bool, "repeat" bool);
CREATE FUNCTION "public"."st_lineinterpolatepoints"("public"."geography", float8, "use_spheroid" bool=true, "repeat" bool=true)
  RETURNS "public"."geography" AS '$libdir/postgis-3', 'geography_line_interpolate_point'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."st_lineinterpolatepoints"("public"."geography", float8, "use_spheroid" bool, "repeat" bool) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_lineinterpolatepoints
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_lineinterpolatepoints"("public"."geometry", float8, "repeat" bool);
CREATE FUNCTION "public"."st_lineinterpolatepoints"("public"."geometry", float8, "repeat" bool=true)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'LWGEOM_line_interpolate_point'
  LANGUAGE c IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_lineinterpolatepoints"("public"."geometry", float8, "repeat" bool) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_linelocatepoint
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_linelocatepoint"("geom1" "public"."geometry", "geom2" "public"."geometry");
CREATE FUNCTION "public"."st_linelocatepoint"("geom1" "public"."geometry", "geom2" "public"."geometry")
  RETURNS "pg_catalog"."float8" AS '$libdir/postgis-3', 'LWGEOM_line_locate_point'
  LANGUAGE c IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_linelocatepoint"("geom1" "public"."geometry", "geom2" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_linelocatepoint
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_linelocatepoint"("public"."geography", "public"."geography", "use_spheroid" bool);
CREATE FUNCTION "public"."st_linelocatepoint"("public"."geography", "public"."geography", "use_spheroid" bool=true)
  RETURNS "pg_catalog"."float8" AS '$libdir/postgis-3', 'geography_line_locate_point'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."st_linelocatepoint"("public"."geography", "public"."geography", "use_spheroid" bool) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_linelocatepoint
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_linelocatepoint"(text, text);
CREATE FUNCTION "public"."st_linelocatepoint"(text, text)
  RETURNS "pg_catalog"."float8" AS $BODY$ SELECT public.ST_LineLocatePoint($1::public.geometry, $2::public.geometry);  $BODY$
  LANGUAGE sql IMMUTABLE
  COST 100;
ALTER FUNCTION "public"."st_linelocatepoint"(text, text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_linemerge
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_linemerge"("public"."geometry", bool);
CREATE FUNCTION "public"."st_linemerge"("public"."geometry", bool)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'linemerge'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_linemerge"("public"."geometry", bool) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_linemerge
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_linemerge"("public"."geometry");
CREATE FUNCTION "public"."st_linemerge"("public"."geometry")
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'linemerge'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_linemerge"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_linestringfromwkb
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_linestringfromwkb"(bytea, int4);
CREATE FUNCTION "public"."st_linestringfromwkb"(bytea, int4)
  RETURNS "public"."geometry" AS $BODY$
	SELECT CASE WHEN public.ST_GeometryType(public.ST_GeomFromWKB($1, $2)) = 'ST_LineString'
	THEN public.ST_GeomFromWKB($1, $2)
	ELSE NULL END
	$BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_linestringfromwkb"(bytea, int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_linestringfromwkb
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_linestringfromwkb"(bytea);
CREATE FUNCTION "public"."st_linestringfromwkb"(bytea)
  RETURNS "public"."geometry" AS $BODY$
	SELECT CASE WHEN public.ST_GeometryType(public.ST_GeomFromWKB($1)) = 'ST_LineString'
	THEN public.ST_GeomFromWKB($1)
	ELSE NULL END
	$BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_linestringfromwkb"(bytea) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_linesubstring
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_linesubstring"(text, float8, float8);
CREATE FUNCTION "public"."st_linesubstring"(text, float8, float8)
  RETURNS "public"."geometry" AS $BODY$ SELECT public.ST_LineSubstring($1::public.geometry, $2, $3);  $BODY$
  LANGUAGE sql IMMUTABLE
  COST 100;
ALTER FUNCTION "public"."st_linesubstring"(text, float8, float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_linesubstring
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_linesubstring"("public"."geography", float8, float8);
CREATE FUNCTION "public"."st_linesubstring"("public"."geography", float8, float8)
  RETURNS "public"."geography" AS '$libdir/postgis-3', 'geography_line_substring'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."st_linesubstring"("public"."geography", float8, float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_linesubstring
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_linesubstring"("public"."geometry", float8, float8);
CREATE FUNCTION "public"."st_linesubstring"("public"."geometry", float8, float8)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'LWGEOM_line_substring'
  LANGUAGE c IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_linesubstring"("public"."geometry", float8, float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_linetocurve
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_linetocurve"("geometry" "public"."geometry");
CREATE FUNCTION "public"."st_linetocurve"("geometry" "public"."geometry")
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'LWGEOM_line_desegmentize'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_linetocurve"("geometry" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_locatealong
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_locatealong"("geometry" "public"."geometry", "measure" float8, "leftrightoffset" float8);
CREATE FUNCTION "public"."st_locatealong"("geometry" "public"."geometry", "measure" float8, "leftrightoffset" float8=0.0)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'ST_LocateAlong'
  LANGUAGE c IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_locatealong"("geometry" "public"."geometry", "measure" float8, "leftrightoffset" float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_locatebetween
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_locatebetween"("geometry" "public"."geometry", "frommeasure" float8, "tomeasure" float8, "leftrightoffset" float8);
CREATE FUNCTION "public"."st_locatebetween"("geometry" "public"."geometry", "frommeasure" float8, "tomeasure" float8, "leftrightoffset" float8=0.0)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'ST_LocateBetween'
  LANGUAGE c IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_locatebetween"("geometry" "public"."geometry", "frommeasure" float8, "tomeasure" float8, "leftrightoffset" float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_locatebetweenelevations
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_locatebetweenelevations"("geometry" "public"."geometry", "fromelevation" float8, "toelevation" float8);
CREATE FUNCTION "public"."st_locatebetweenelevations"("geometry" "public"."geometry", "fromelevation" float8, "toelevation" float8)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'ST_LocateBetweenElevations'
  LANGUAGE c IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_locatebetweenelevations"("geometry" "public"."geometry", "fromelevation" float8, "toelevation" float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_longestline
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_longestline"("geom1" "public"."geometry", "geom2" "public"."geometry");
CREATE FUNCTION "public"."st_longestline"("geom1" "public"."geometry", "geom2" "public"."geometry")
  RETURNS "public"."geometry" AS $BODY$SELECT public._ST_LongestLine(public.ST_ConvexHull($1), public.ST_ConvexHull($2))$BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_longestline"("geom1" "public"."geometry", "geom2" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_m
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_m"("public"."geometry");
CREATE FUNCTION "public"."st_m"("public"."geometry")
  RETURNS "pg_catalog"."float8" AS '$libdir/postgis-3', 'LWGEOM_m_point'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."st_m"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_makebox2d
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_makebox2d"("geom1" "public"."geometry", "geom2" "public"."geometry");
CREATE FUNCTION "public"."st_makebox2d"("geom1" "public"."geometry", "geom2" "public"."geometry")
  RETURNS "public"."box2d" AS '$libdir/postgis-3', 'BOX2D_construct'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."st_makebox2d"("geom1" "public"."geometry", "geom2" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_makeenvelope
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_makeenvelope"(float8, float8, float8, float8, int4);
CREATE FUNCTION "public"."st_makeenvelope"(float8, float8, float8, float8, int4=0)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'ST_MakeEnvelope'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_makeenvelope"(float8, float8, float8, float8, int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_makeline
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_makeline"("geom1" "public"."geometry", "geom2" "public"."geometry");
CREATE FUNCTION "public"."st_makeline"("geom1" "public"."geometry", "geom2" "public"."geometry")
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'LWGEOM_makeline'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_makeline"("geom1" "public"."geometry", "geom2" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_makeline
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_makeline"("public"."_geometry");
CREATE FUNCTION "public"."st_makeline"("public"."_geometry")
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'LWGEOM_makeline_garray'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_makeline"("public"."_geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_makepoint
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_makepoint"(float8, float8, float8);
CREATE FUNCTION "public"."st_makepoint"(float8, float8, float8)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'LWGEOM_makepoint'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_makepoint"(float8, float8, float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_makepoint
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_makepoint"(float8, float8, float8, float8);
CREATE FUNCTION "public"."st_makepoint"(float8, float8, float8, float8)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'LWGEOM_makepoint'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_makepoint"(float8, float8, float8, float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_makepoint
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_makepoint"(float8, float8);
CREATE FUNCTION "public"."st_makepoint"(float8, float8)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'LWGEOM_makepoint'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_makepoint"(float8, float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_makepointm
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_makepointm"(float8, float8, float8);
CREATE FUNCTION "public"."st_makepointm"(float8, float8, float8)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'LWGEOM_makepoint3dm'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_makepointm"(float8, float8, float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_makepolygon
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_makepolygon"("public"."geometry");
CREATE FUNCTION "public"."st_makepolygon"("public"."geometry")
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'LWGEOM_makepoly'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_makepolygon"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_makepolygon
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_makepolygon"("public"."geometry", "public"."_geometry");
CREATE FUNCTION "public"."st_makepolygon"("public"."geometry", "public"."_geometry")
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'LWGEOM_makepoly'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_makepolygon"("public"."geometry", "public"."_geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_makevalid
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_makevalid"("public"."geometry");
CREATE FUNCTION "public"."st_makevalid"("public"."geometry")
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'ST_MakeValid'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_makevalid"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_makevalid
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_makevalid"("geom" "public"."geometry", "params" text);
CREATE FUNCTION "public"."st_makevalid"("geom" "public"."geometry", "params" text)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'ST_MakeValid'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_makevalid"("geom" "public"."geometry", "params" text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_maxdistance
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_maxdistance"("geom1" "public"."geometry", "geom2" "public"."geometry");
CREATE FUNCTION "public"."st_maxdistance"("geom1" "public"."geometry", "geom2" "public"."geometry")
  RETURNS "pg_catalog"."float8" AS $BODY$SELECT public._ST_MaxDistance(public.ST_ConvexHull($1), public.ST_ConvexHull($2))$BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_maxdistance"("geom1" "public"."geometry", "geom2" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_maximuminscribedcircle
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_maximuminscribedcircle"("public"."geometry", OUT "center" "public"."geometry", OUT "nearest" "public"."geometry", OUT "radius" float8);
CREATE FUNCTION "public"."st_maximuminscribedcircle"(IN "public"."geometry", OUT "center" "public"."geometry", OUT "nearest" "public"."geometry", OUT "radius" float8)
  RETURNS "pg_catalog"."record" AS '$libdir/postgis-3', 'ST_MaximumInscribedCircle'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_maximuminscribedcircle"("public"."geometry", OUT "center" "public"."geometry", OUT "nearest" "public"."geometry", OUT "radius" float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_memsize
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_memsize"("public"."geometry");
CREATE FUNCTION "public"."st_memsize"("public"."geometry")
  RETURNS "pg_catalog"."int4" AS '$libdir/postgis-3', 'LWGEOM_mem_size'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."st_memsize"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_minimumboundingcircle
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_minimumboundingcircle"("inputgeom" "public"."geometry", "segs_per_quarter" int4);
CREATE FUNCTION "public"."st_minimumboundingcircle"("inputgeom" "public"."geometry", "segs_per_quarter" int4=48)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'ST_MinimumBoundingCircle'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_minimumboundingcircle"("inputgeom" "public"."geometry", "segs_per_quarter" int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_minimumboundingradius
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_minimumboundingradius"("public"."geometry", OUT "center" "public"."geometry", OUT "radius" float8);
CREATE FUNCTION "public"."st_minimumboundingradius"(IN "public"."geometry", OUT "center" "public"."geometry", OUT "radius" float8)
  RETURNS "pg_catalog"."record" AS '$libdir/postgis-3', 'ST_MinimumBoundingRadius'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_minimumboundingradius"("public"."geometry", OUT "center" "public"."geometry", OUT "radius" float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_minimumclearance
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_minimumclearance"("public"."geometry");
CREATE FUNCTION "public"."st_minimumclearance"("public"."geometry")
  RETURNS "pg_catalog"."float8" AS '$libdir/postgis-3', 'ST_MinimumClearance'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_minimumclearance"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_minimumclearanceline
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_minimumclearanceline"("public"."geometry");
CREATE FUNCTION "public"."st_minimumclearanceline"("public"."geometry")
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'ST_MinimumClearanceLine'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_minimumclearanceline"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_mlinefromtext
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_mlinefromtext"(text, int4);
CREATE FUNCTION "public"."st_mlinefromtext"(text, int4)
  RETURNS "public"."geometry" AS $BODY$
	SELECT CASE
	WHEN public.ST_GeometryType(public.ST_GeomFromText($1, $2)) = 'ST_MultiLineString'
	THEN public.ST_GeomFromText($1,$2)
	ELSE NULL END
	$BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_mlinefromtext"(text, int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_mlinefromtext
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_mlinefromtext"(text);
CREATE FUNCTION "public"."st_mlinefromtext"(text)
  RETURNS "public"."geometry" AS $BODY$
	SELECT CASE WHEN public.ST_GeometryType(public.ST_GeomFromText($1)) = 'ST_MultiLineString'
	THEN public.ST_GeomFromText($1)
	ELSE NULL END
	$BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_mlinefromtext"(text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_mlinefromwkb
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_mlinefromwkb"(bytea, int4);
CREATE FUNCTION "public"."st_mlinefromwkb"(bytea, int4)
  RETURNS "public"."geometry" AS $BODY$
	SELECT CASE WHEN public.ST_GeometryType(public.ST_GeomFromWKB($1, $2)) = 'ST_MultiLineString'
	THEN public.ST_GeomFromWKB($1, $2)
	ELSE NULL END
	$BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_mlinefromwkb"(bytea, int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_mlinefromwkb
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_mlinefromwkb"(bytea);
CREATE FUNCTION "public"."st_mlinefromwkb"(bytea)
  RETURNS "public"."geometry" AS $BODY$
	SELECT CASE WHEN public.ST_GeometryType(public.ST_GeomFromWKB($1)) = 'ST_MultiLineString'
	THEN public.ST_GeomFromWKB($1)
	ELSE NULL END
	$BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_mlinefromwkb"(bytea) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_mpointfromtext
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_mpointfromtext"(text);
CREATE FUNCTION "public"."st_mpointfromtext"(text)
  RETURNS "public"."geometry" AS $BODY$
	SELECT CASE WHEN public.ST_GeometryType(public.ST_GeomFromText($1)) = 'ST_MultiPoint'
	THEN public.ST_GeomFromText($1)
	ELSE NULL END
	$BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_mpointfromtext"(text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_mpointfromtext
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_mpointfromtext"(text, int4);
CREATE FUNCTION "public"."st_mpointfromtext"(text, int4)
  RETURNS "public"."geometry" AS $BODY$
	SELECT CASE WHEN public.ST_GeometryType(public.ST_GeomFromText($1, $2)) = 'ST_MultiPoint'
	THEN ST_GeomFromText($1, $2)
	ELSE NULL END
	$BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_mpointfromtext"(text, int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_mpointfromwkb
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_mpointfromwkb"(bytea);
CREATE FUNCTION "public"."st_mpointfromwkb"(bytea)
  RETURNS "public"."geometry" AS $BODY$
	SELECT CASE WHEN public.ST_GeometryType(public.ST_GeomFromWKB($1)) = 'ST_MultiPoint'
	THEN public.ST_GeomFromWKB($1)
	ELSE NULL END
	$BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_mpointfromwkb"(bytea) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_mpointfromwkb
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_mpointfromwkb"(bytea, int4);
CREATE FUNCTION "public"."st_mpointfromwkb"(bytea, int4)
  RETURNS "public"."geometry" AS $BODY$
	SELECT CASE WHEN public.ST_GeometryType(public.ST_GeomFromWKB($1, $2)) = 'ST_MultiPoint'
	THEN public.ST_GeomFromWKB($1, $2)
	ELSE NULL END
	$BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_mpointfromwkb"(bytea, int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_mpolyfromtext
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_mpolyfromtext"(text);
CREATE FUNCTION "public"."st_mpolyfromtext"(text)
  RETURNS "public"."geometry" AS $BODY$
	SELECT CASE WHEN public.ST_GeometryType(public.ST_GeomFromText($1)) = 'ST_MultiPolygon'
	THEN public.ST_GeomFromText($1)
	ELSE NULL END
	$BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_mpolyfromtext"(text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_mpolyfromtext
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_mpolyfromtext"(text, int4);
CREATE FUNCTION "public"."st_mpolyfromtext"(text, int4)
  RETURNS "public"."geometry" AS $BODY$
	SELECT CASE WHEN public.ST_GeometryType(public.ST_GeomFromText($1, $2)) = 'ST_MultiPolygon'
	THEN public.ST_GeomFromText($1,$2)
	ELSE NULL END
	$BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_mpolyfromtext"(text, int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_mpolyfromwkb
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_mpolyfromwkb"(bytea);
CREATE FUNCTION "public"."st_mpolyfromwkb"(bytea)
  RETURNS "public"."geometry" AS $BODY$
	SELECT CASE WHEN public.ST_GeometryType(public.ST_GeomFromWKB($1)) = 'ST_MultiPolygon'
	THEN public.ST_GeomFromWKB($1)
	ELSE NULL END
	$BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 100;
ALTER FUNCTION "public"."st_mpolyfromwkb"(bytea) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_mpolyfromwkb
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_mpolyfromwkb"(bytea, int4);
CREATE FUNCTION "public"."st_mpolyfromwkb"(bytea, int4)
  RETURNS "public"."geometry" AS $BODY$
	SELECT CASE WHEN public.ST_GeometryType(public.ST_GeomFromWKB($1, $2)) = 'ST_MultiPolygon'
	THEN public.ST_GeomFromWKB($1, $2)
	ELSE NULL END
	$BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_mpolyfromwkb"(bytea, int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_multi
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_multi"("public"."geometry");
CREATE FUNCTION "public"."st_multi"("public"."geometry")
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'LWGEOM_force_multi'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_multi"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_multilinefromwkb
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_multilinefromwkb"(bytea);
CREATE FUNCTION "public"."st_multilinefromwkb"(bytea)
  RETURNS "public"."geometry" AS $BODY$
	SELECT CASE WHEN public.ST_GeometryType(public.ST_GeomFromWKB($1)) = 'ST_MultiLineString'
	THEN public.ST_GeomFromWKB($1)
	ELSE NULL END
	$BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_multilinefromwkb"(bytea) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_multilinestringfromtext
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_multilinestringfromtext"(text);
CREATE FUNCTION "public"."st_multilinestringfromtext"(text)
  RETURNS "public"."geometry" AS $BODY$SELECT public.ST_MLineFromText($1)$BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_multilinestringfromtext"(text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_multilinestringfromtext
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_multilinestringfromtext"(text, int4);
CREATE FUNCTION "public"."st_multilinestringfromtext"(text, int4)
  RETURNS "public"."geometry" AS $BODY$SELECT public.ST_MLineFromText($1, $2)$BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_multilinestringfromtext"(text, int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_multipointfromtext
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_multipointfromtext"(text);
CREATE FUNCTION "public"."st_multipointfromtext"(text)
  RETURNS "public"."geometry" AS $BODY$SELECT public.ST_MPointFromText($1)$BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_multipointfromtext"(text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_multipointfromwkb
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_multipointfromwkb"(bytea, int4);
CREATE FUNCTION "public"."st_multipointfromwkb"(bytea, int4)
  RETURNS "public"."geometry" AS $BODY$
	SELECT CASE WHEN public.ST_GeometryType(public.ST_GeomFromWKB($1,$2)) = 'ST_MultiPoint'
	THEN public.ST_GeomFromWKB($1, $2)
	ELSE NULL END
	$BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_multipointfromwkb"(bytea, int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_multipointfromwkb
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_multipointfromwkb"(bytea);
CREATE FUNCTION "public"."st_multipointfromwkb"(bytea)
  RETURNS "public"."geometry" AS $BODY$
	SELECT CASE WHEN public.ST_GeometryType(public.ST_GeomFromWKB($1)) = 'ST_MultiPoint'
	THEN public.ST_GeomFromWKB($1)
	ELSE NULL END
	$BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_multipointfromwkb"(bytea) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_multipolyfromwkb
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_multipolyfromwkb"(bytea, int4);
CREATE FUNCTION "public"."st_multipolyfromwkb"(bytea, int4)
  RETURNS "public"."geometry" AS $BODY$
	SELECT CASE WHEN public.ST_GeometryType(public.ST_GeomFromWKB($1, $2)) = 'ST_MultiPolygon'
	THEN public.ST_GeomFromWKB($1, $2)
	ELSE NULL END
	$BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_multipolyfromwkb"(bytea, int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_multipolyfromwkb
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_multipolyfromwkb"(bytea);
CREATE FUNCTION "public"."st_multipolyfromwkb"(bytea)
  RETURNS "public"."geometry" AS $BODY$
	SELECT CASE WHEN public.ST_GeometryType(public.ST_GeomFromWKB($1)) = 'ST_MultiPolygon'
	THEN public.ST_GeomFromWKB($1)
	ELSE NULL END
	$BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_multipolyfromwkb"(bytea) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_multipolygonfromtext
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_multipolygonfromtext"(text);
CREATE FUNCTION "public"."st_multipolygonfromtext"(text)
  RETURNS "public"."geometry" AS $BODY$SELECT public.ST_MPolyFromText($1)$BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_multipolygonfromtext"(text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_multipolygonfromtext
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_multipolygonfromtext"(text, int4);
CREATE FUNCTION "public"."st_multipolygonfromtext"(text, int4)
  RETURNS "public"."geometry" AS $BODY$SELECT public.ST_MPolyFromText($1, $2)$BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_multipolygonfromtext"(text, int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_ndims
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_ndims"("public"."geometry");
CREATE FUNCTION "public"."st_ndims"("public"."geometry")
  RETURNS "pg_catalog"."int2" AS '$libdir/postgis-3', 'LWGEOM_ndims'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."st_ndims"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_node
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_node"("g" "public"."geometry");
CREATE FUNCTION "public"."st_node"("g" "public"."geometry")
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'ST_Node'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_node"("g" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_normalize
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_normalize"("geom" "public"."geometry");
CREATE FUNCTION "public"."st_normalize"("geom" "public"."geometry")
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'ST_Normalize'
  LANGUAGE c IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_normalize"("geom" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_npoints
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_npoints"("public"."geometry");
CREATE FUNCTION "public"."st_npoints"("public"."geometry")
  RETURNS "pg_catalog"."int4" AS '$libdir/postgis-3', 'LWGEOM_npoints'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_npoints"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_nrings
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_nrings"("public"."geometry");
CREATE FUNCTION "public"."st_nrings"("public"."geometry")
  RETURNS "pg_catalog"."int4" AS '$libdir/postgis-3', 'LWGEOM_nrings'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_nrings"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_numcurves
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_numcurves"("geometry" "public"."geometry");
CREATE FUNCTION "public"."st_numcurves"("geometry" "public"."geometry")
  RETURNS "pg_catalog"."int4" AS '$libdir/postgis-3', 'ST_NumCurves'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_numcurves"("geometry" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_numgeometries
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_numgeometries"("public"."geometry");
CREATE FUNCTION "public"."st_numgeometries"("public"."geometry")
  RETURNS "pg_catalog"."int4" AS '$libdir/postgis-3', 'LWGEOM_numgeometries_collection'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_numgeometries"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_numinteriorring
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_numinteriorring"("public"."geometry");
CREATE FUNCTION "public"."st_numinteriorring"("public"."geometry")
  RETURNS "pg_catalog"."int4" AS '$libdir/postgis-3', 'LWGEOM_numinteriorrings_polygon'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_numinteriorring"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_numinteriorrings
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_numinteriorrings"("public"."geometry");
CREATE FUNCTION "public"."st_numinteriorrings"("public"."geometry")
  RETURNS "pg_catalog"."int4" AS '$libdir/postgis-3', 'LWGEOM_numinteriorrings_polygon'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_numinteriorrings"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_numpatches
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_numpatches"("public"."geometry");
CREATE FUNCTION "public"."st_numpatches"("public"."geometry")
  RETURNS "pg_catalog"."int4" AS '$libdir/postgis-3', 'LWGEOM_numpatches'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_numpatches"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_numpoints
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_numpoints"("public"."geometry");
CREATE FUNCTION "public"."st_numpoints"("public"."geometry")
  RETURNS "pg_catalog"."int4" AS '$libdir/postgis-3', 'LWGEOM_numpoints_linestring'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_numpoints"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_offsetcurve
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_offsetcurve"("line" "public"."geometry", "distance" float8, "params" text);
CREATE FUNCTION "public"."st_offsetcurve"("line" "public"."geometry", "distance" float8, "params" text=''::text)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'ST_OffsetCurve'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_offsetcurve"("line" "public"."geometry", "distance" float8, "params" text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_orderingequals
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_orderingequals"("geom1" "public"."geometry", "geom2" "public"."geometry");
CREATE FUNCTION "public"."st_orderingequals"("geom1" "public"."geometry", "geom2" "public"."geometry")
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'LWGEOM_same'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_orderingequals"("geom1" "public"."geometry", "geom2" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_orientedenvelope
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_orientedenvelope"("public"."geometry");
CREATE FUNCTION "public"."st_orientedenvelope"("public"."geometry")
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'ST_OrientedEnvelope'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_orientedenvelope"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_overlaps
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_overlaps"("geom1" "public"."geometry", "geom2" "public"."geometry");
CREATE FUNCTION "public"."st_overlaps"("geom1" "public"."geometry", "geom2" "public"."geometry")
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'overlaps'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_overlaps"("geom1" "public"."geometry", "geom2" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_patchn
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_patchn"("public"."geometry", int4);
CREATE FUNCTION "public"."st_patchn"("public"."geometry", int4)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'LWGEOM_patchn'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."st_patchn"("public"."geometry", int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_perimeter
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_perimeter"("geog" "public"."geography", "use_spheroid" bool);
CREATE FUNCTION "public"."st_perimeter"("geog" "public"."geography", "use_spheroid" bool=true)
  RETURNS "pg_catalog"."float8" AS '$libdir/postgis-3', 'geography_perimeter'
  LANGUAGE c IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_perimeter"("geog" "public"."geography", "use_spheroid" bool) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_perimeter
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_perimeter"("public"."geometry");
CREATE FUNCTION "public"."st_perimeter"("public"."geometry")
  RETURNS "pg_catalog"."float8" AS '$libdir/postgis-3', 'LWGEOM_perimeter2d_poly'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_perimeter"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_perimeter2d
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_perimeter2d"("public"."geometry");
CREATE FUNCTION "public"."st_perimeter2d"("public"."geometry")
  RETURNS "pg_catalog"."float8" AS '$libdir/postgis-3', 'LWGEOM_perimeter2d_poly'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_perimeter2d"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_point
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_point"(float8, float8);
CREATE FUNCTION "public"."st_point"(float8, float8)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'LWGEOM_makepoint'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_point"(float8, float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_point
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_point"(float8, float8, "srid" int4);
CREATE FUNCTION "public"."st_point"(float8, float8, "srid" int4)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'ST_Point'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_point"(float8, float8, "srid" int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_pointfromgeohash
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_pointfromgeohash"(text, int4);
CREATE FUNCTION "public"."st_pointfromgeohash"(text, int4=NULL::integer)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'point_from_geohash'
  LANGUAGE c IMMUTABLE
  COST 50;
ALTER FUNCTION "public"."st_pointfromgeohash"(text, int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_pointfromtext
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_pointfromtext"(text, int4);
CREATE FUNCTION "public"."st_pointfromtext"(text, int4)
  RETURNS "public"."geometry" AS $BODY$
	SELECT CASE WHEN public.ST_GeometryType(public.ST_GeomFromText($1, $2)) = 'ST_Point'
	THEN public.ST_GeomFromText($1, $2)
	ELSE NULL END
	$BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_pointfromtext"(text, int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_pointfromtext
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_pointfromtext"(text);
CREATE FUNCTION "public"."st_pointfromtext"(text)
  RETURNS "public"."geometry" AS $BODY$
	SELECT CASE WHEN public.ST_GeometryType(public.ST_GeomFromText($1)) = 'ST_Point'
	THEN public.ST_GeomFromText($1)
	ELSE NULL END
	$BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_pointfromtext"(text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_pointfromwkb
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_pointfromwkb"(bytea, int4);
CREATE FUNCTION "public"."st_pointfromwkb"(bytea, int4)
  RETURNS "public"."geometry" AS $BODY$
	SELECT CASE WHEN public.ST_GeometryType(public.ST_GeomFromWKB($1, $2)) = 'ST_Point'
	THEN public.ST_GeomFromWKB($1, $2)
	ELSE NULL END
	$BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_pointfromwkb"(bytea, int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_pointfromwkb
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_pointfromwkb"(bytea);
CREATE FUNCTION "public"."st_pointfromwkb"(bytea)
  RETURNS "public"."geometry" AS $BODY$
	SELECT CASE WHEN public.ST_GeometryType(public.ST_GeomFromWKB($1)) = 'ST_Point'
	THEN public.ST_GeomFromWKB($1)
	ELSE NULL END
	$BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_pointfromwkb"(bytea) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_pointinsidecircle
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_pointinsidecircle"("public"."geometry", float8, float8, float8);
CREATE FUNCTION "public"."st_pointinsidecircle"("public"."geometry", float8, float8, float8)
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'LWGEOM_inside_circle_point'
  LANGUAGE c IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_pointinsidecircle"("public"."geometry", float8, float8, float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_pointm
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_pointm"("xcoordinate" float8, "ycoordinate" float8, "mcoordinate" float8, "srid" int4);
CREATE FUNCTION "public"."st_pointm"("xcoordinate" float8, "ycoordinate" float8, "mcoordinate" float8, "srid" int4=0)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'ST_PointM'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_pointm"("xcoordinate" float8, "ycoordinate" float8, "mcoordinate" float8, "srid" int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_pointn
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_pointn"("public"."geometry", int4);
CREATE FUNCTION "public"."st_pointn"("public"."geometry", int4)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'LWGEOM_pointn_linestring'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_pointn"("public"."geometry", int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_pointonsurface
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_pointonsurface"("public"."geometry");
CREATE FUNCTION "public"."st_pointonsurface"("public"."geometry")
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'pointonsurface'
  LANGUAGE c IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_pointonsurface"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_points
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_points"("public"."geometry");
CREATE FUNCTION "public"."st_points"("public"."geometry")
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'ST_Points'
  LANGUAGE c IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_points"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_pointz
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_pointz"("xcoordinate" float8, "ycoordinate" float8, "zcoordinate" float8, "srid" int4);
CREATE FUNCTION "public"."st_pointz"("xcoordinate" float8, "ycoordinate" float8, "zcoordinate" float8, "srid" int4=0)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'ST_PointZ'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_pointz"("xcoordinate" float8, "ycoordinate" float8, "zcoordinate" float8, "srid" int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_pointzm
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_pointzm"("xcoordinate" float8, "ycoordinate" float8, "zcoordinate" float8, "mcoordinate" float8, "srid" int4);
CREATE FUNCTION "public"."st_pointzm"("xcoordinate" float8, "ycoordinate" float8, "zcoordinate" float8, "mcoordinate" float8, "srid" int4=0)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'ST_PointZM'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_pointzm"("xcoordinate" float8, "ycoordinate" float8, "zcoordinate" float8, "mcoordinate" float8, "srid" int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_polyfromtext
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_polyfromtext"(text);
CREATE FUNCTION "public"."st_polyfromtext"(text)
  RETURNS "public"."geometry" AS $BODY$
	SELECT CASE WHEN public.ST_GeometryType(public.ST_GeomFromText($1)) = 'ST_Polygon'
	THEN public.ST_GeomFromText($1)
	ELSE NULL END
	$BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_polyfromtext"(text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_polyfromtext
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_polyfromtext"(text, int4);
CREATE FUNCTION "public"."st_polyfromtext"(text, int4)
  RETURNS "public"."geometry" AS $BODY$
	SELECT CASE WHEN public.ST_GeometryType(public.ST_GeomFromText($1, $2)) = 'ST_Polygon'
	THEN public.ST_GeomFromText($1, $2)
	ELSE NULL END
	$BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_polyfromtext"(text, int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_polyfromwkb
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_polyfromwkb"(bytea);
CREATE FUNCTION "public"."st_polyfromwkb"(bytea)
  RETURNS "public"."geometry" AS $BODY$
	SELECT CASE WHEN public.ST_GeometryType(public.ST_GeomFromWKB($1)) = 'ST_Polygon'
	THEN public.ST_GeomFromWKB($1)
	ELSE NULL END
	$BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_polyfromwkb"(bytea) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_polyfromwkb
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_polyfromwkb"(bytea, int4);
CREATE FUNCTION "public"."st_polyfromwkb"(bytea, int4)
  RETURNS "public"."geometry" AS $BODY$
	SELECT CASE WHEN public.ST_GeometryType(public.ST_GeomFromWKB($1, $2)) = 'ST_Polygon'
	THEN public.ST_GeomFromWKB($1, $2)
	ELSE NULL END
	$BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_polyfromwkb"(bytea, int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_polygon
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_polygon"("public"."geometry", int4);
CREATE FUNCTION "public"."st_polygon"("public"."geometry", int4)
  RETURNS "public"."geometry" AS $BODY$
	SELECT public.ST_SetSRID(public.ST_MakePolygon($1), $2)
	$BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_polygon"("public"."geometry", int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_polygonfromtext
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_polygonfromtext"(text, int4);
CREATE FUNCTION "public"."st_polygonfromtext"(text, int4)
  RETURNS "public"."geometry" AS $BODY$SELECT public.ST_PolyFromText($1, $2)$BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_polygonfromtext"(text, int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_polygonfromtext
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_polygonfromtext"(text);
CREATE FUNCTION "public"."st_polygonfromtext"(text)
  RETURNS "public"."geometry" AS $BODY$SELECT public.ST_PolyFromText($1)$BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_polygonfromtext"(text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_polygonfromwkb
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_polygonfromwkb"(bytea, int4);
CREATE FUNCTION "public"."st_polygonfromwkb"(bytea, int4)
  RETURNS "public"."geometry" AS $BODY$
	SELECT CASE WHEN public.ST_GeometryType(public.ST_GeomFromWKB($1,$2)) = 'ST_Polygon'
	THEN public.ST_GeomFromWKB($1, $2)
	ELSE NULL END
	$BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_polygonfromwkb"(bytea, int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_polygonfromwkb
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_polygonfromwkb"(bytea);
CREATE FUNCTION "public"."st_polygonfromwkb"(bytea)
  RETURNS "public"."geometry" AS $BODY$
	SELECT CASE WHEN public.ST_GeometryType(public.ST_GeomFromWKB($1)) = 'ST_Polygon'
	THEN public.ST_GeomFromWKB($1)
	ELSE NULL END
	$BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_polygonfromwkb"(bytea) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_polygonize
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_polygonize"("public"."_geometry");
CREATE FUNCTION "public"."st_polygonize"("public"."_geometry")
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'polygonize_garray'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_polygonize"("public"."_geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_project
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_project"("geog_from" "public"."geography", "geog_to" "public"."geography", "distance" float8);
CREATE FUNCTION "public"."st_project"("geog_from" "public"."geography", "geog_to" "public"."geography", "distance" float8)
  RETURNS "public"."geography" AS '$libdir/postgis-3', 'geography_project_geography'
  LANGUAGE c IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_project"("geog_from" "public"."geography", "geog_to" "public"."geography", "distance" float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_project
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_project"("geog" "public"."geography", "distance" float8, "azimuth" float8);
CREATE FUNCTION "public"."st_project"("geog" "public"."geography", "distance" float8, "azimuth" float8)
  RETURNS "public"."geography" AS '$libdir/postgis-3', 'geography_project'
  LANGUAGE c IMMUTABLE
  COST 250;
ALTER FUNCTION "public"."st_project"("geog" "public"."geography", "distance" float8, "azimuth" float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_project
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_project"("geom1" "public"."geometry", "geom2" "public"."geometry", "distance" float8);
CREATE FUNCTION "public"."st_project"("geom1" "public"."geometry", "geom2" "public"."geometry", "distance" float8)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'geometry_project_geometry'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_project"("geom1" "public"."geometry", "geom2" "public"."geometry", "distance" float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_project
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_project"("geom1" "public"."geometry", "distance" float8, "azimuth" float8);
CREATE FUNCTION "public"."st_project"("geom1" "public"."geometry", "distance" float8, "azimuth" float8)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'geometry_project_direction'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_project"("geom1" "public"."geometry", "distance" float8, "azimuth" float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_quantizecoordinates
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_quantizecoordinates"("g" "public"."geometry", "prec_x" int4, "prec_y" int4, "prec_z" int4, "prec_m" int4);
CREATE FUNCTION "public"."st_quantizecoordinates"("g" "public"."geometry", "prec_x" int4, "prec_y" int4=NULL::integer, "prec_z" int4=NULL::integer, "prec_m" int4=NULL::integer)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'ST_QuantizeCoordinates'
  LANGUAGE c IMMUTABLE
  COST 250;
ALTER FUNCTION "public"."st_quantizecoordinates"("g" "public"."geometry", "prec_x" int4, "prec_y" int4, "prec_z" int4, "prec_m" int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_reduceprecision
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_reduceprecision"("geom" "public"."geometry", "gridsize" float8);
CREATE FUNCTION "public"."st_reduceprecision"("geom" "public"."geometry", "gridsize" float8)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'ST_ReducePrecision'
  LANGUAGE c IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_reduceprecision"("geom" "public"."geometry", "gridsize" float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_relate
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_relate"("geom1" "public"."geometry", "geom2" "public"."geometry");
CREATE FUNCTION "public"."st_relate"("geom1" "public"."geometry", "geom2" "public"."geometry")
  RETURNS "pg_catalog"."text" AS '$libdir/postgis-3', 'relate_full'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_relate"("geom1" "public"."geometry", "geom2" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_relate
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_relate"("geom1" "public"."geometry", "geom2" "public"."geometry", text);
CREATE FUNCTION "public"."st_relate"("geom1" "public"."geometry", "geom2" "public"."geometry", text)
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'relate_pattern'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_relate"("geom1" "public"."geometry", "geom2" "public"."geometry", text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_relate
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_relate"("geom1" "public"."geometry", "geom2" "public"."geometry", int4);
CREATE FUNCTION "public"."st_relate"("geom1" "public"."geometry", "geom2" "public"."geometry", int4)
  RETURNS "pg_catalog"."text" AS '$libdir/postgis-3', 'relate_full'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_relate"("geom1" "public"."geometry", "geom2" "public"."geometry", int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_relatematch
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_relatematch"(text, text);
CREATE FUNCTION "public"."st_relatematch"(text, text)
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'ST_RelateMatch'
  LANGUAGE c IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_relatematch"(text, text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_removeirrelevantpointsforview
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_removeirrelevantpointsforview"("public"."geometry", "public"."box2d", bool);
CREATE FUNCTION "public"."st_removeirrelevantpointsforview"("public"."geometry", "public"."box2d", bool=false)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'ST_RemoveIrrelevantPointsForView'
  LANGUAGE c IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_removeirrelevantpointsforview"("public"."geometry", "public"."box2d", bool) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_removepoint
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_removepoint"("public"."geometry", int4);
CREATE FUNCTION "public"."st_removepoint"("public"."geometry", int4)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'LWGEOM_removepoint'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_removepoint"("public"."geometry", int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_removerepeatedpoints
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_removerepeatedpoints"("geom" "public"."geometry", "tolerance" float8);
CREATE FUNCTION "public"."st_removerepeatedpoints"("geom" "public"."geometry", "tolerance" float8=0.0)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'ST_RemoveRepeatedPoints'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_removerepeatedpoints"("geom" "public"."geometry", "tolerance" float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_removesmallparts
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_removesmallparts"("public"."geometry", float8, float8);
CREATE FUNCTION "public"."st_removesmallparts"("public"."geometry", float8, float8)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'ST_RemoveSmallParts'
  LANGUAGE c IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_removesmallparts"("public"."geometry", float8, float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_reverse
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_reverse"("public"."geometry");
CREATE FUNCTION "public"."st_reverse"("public"."geometry")
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'LWGEOM_reverse'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_reverse"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_rotate
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_rotate"("public"."geometry", float8, float8, float8);
CREATE FUNCTION "public"."st_rotate"("public"."geometry", float8, float8, float8)
  RETURNS "public"."geometry" AS $BODY$SELECT public.ST_Affine($1,  cos($2), -sin($2), 0,  sin($2),  cos($2), 0, 0, 0, 1,	$3 - cos($2) * $3 + sin($2) * $4, $4 - sin($2) * $3 - cos($2) * $4, 0)$BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_rotate"("public"."geometry", float8, float8, float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_rotate
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_rotate"("public"."geometry", float8, "public"."geometry");
CREATE FUNCTION "public"."st_rotate"("public"."geometry", float8, "public"."geometry")
  RETURNS "public"."geometry" AS $BODY$SELECT public.ST_Affine($1,  cos($2), -sin($2), 0,  sin($2),  cos($2), 0, 0, 0, 1, public.ST_X($3) - cos($2) * public.ST_X($3) + sin($2) * public.ST_Y($3), public.ST_Y($3) - sin($2) * public.ST_X($3) - cos($2) * public.ST_Y($3), 0)$BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_rotate"("public"."geometry", float8, "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_rotate
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_rotate"("public"."geometry", float8);
CREATE FUNCTION "public"."st_rotate"("public"."geometry", float8)
  RETURNS "public"."geometry" AS $BODY$SELECT public.ST_Affine($1,  cos($2), -sin($2), 0,  sin($2), cos($2), 0,  0, 0, 1,  0, 0, 0)$BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_rotate"("public"."geometry", float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_rotatex
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_rotatex"("public"."geometry", float8);
CREATE FUNCTION "public"."st_rotatex"("public"."geometry", float8)
  RETURNS "public"."geometry" AS $BODY$SELECT public.ST_Affine($1, 1, 0, 0, 0, cos($2), -sin($2), 0, sin($2), cos($2), 0, 0, 0)$BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_rotatex"("public"."geometry", float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_rotatey
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_rotatey"("public"."geometry", float8);
CREATE FUNCTION "public"."st_rotatey"("public"."geometry", float8)
  RETURNS "public"."geometry" AS $BODY$SELECT public.ST_Affine($1,  cos($2), 0, sin($2),  0, 1, 0,  -sin($2), 0, cos($2), 0,  0, 0)$BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_rotatey"("public"."geometry", float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_rotatez
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_rotatez"("public"."geometry", float8);
CREATE FUNCTION "public"."st_rotatez"("public"."geometry", float8)
  RETURNS "public"."geometry" AS $BODY$SELECT public.ST_Rotate($1, $2)$BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_rotatez"("public"."geometry", float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_scale
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_scale"("public"."geometry", float8, float8);
CREATE FUNCTION "public"."st_scale"("public"."geometry", float8, float8)
  RETURNS "public"."geometry" AS $BODY$SELECT public.ST_Scale($1, $2, $3, 1)$BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_scale"("public"."geometry", float8, float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_scale
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_scale"("public"."geometry", float8, float8, float8);
CREATE FUNCTION "public"."st_scale"("public"."geometry", float8, float8, float8)
  RETURNS "public"."geometry" AS $BODY$SELECT public.ST_Scale($1, public.ST_MakePoint($2, $3, $4))$BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_scale"("public"."geometry", float8, float8, float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_scale
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_scale"("public"."geometry", "public"."geometry", "origin" "public"."geometry");
CREATE FUNCTION "public"."st_scale"("public"."geometry", "public"."geometry", "origin" "public"."geometry")
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'ST_Scale'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_scale"("public"."geometry", "public"."geometry", "origin" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_scale
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_scale"("public"."geometry", "public"."geometry");
CREATE FUNCTION "public"."st_scale"("public"."geometry", "public"."geometry")
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'ST_Scale'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_scale"("public"."geometry", "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_scroll
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_scroll"("public"."geometry", "public"."geometry");
CREATE FUNCTION "public"."st_scroll"("public"."geometry", "public"."geometry")
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'ST_Scroll'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_scroll"("public"."geometry", "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_segmentize
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_segmentize"("public"."geometry", float8);
CREATE FUNCTION "public"."st_segmentize"("public"."geometry", float8)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'LWGEOM_segmentize2d'
  LANGUAGE c IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_segmentize"("public"."geometry", float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_segmentize
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_segmentize"("geog" "public"."geography", "max_segment_length" float8);
CREATE FUNCTION "public"."st_segmentize"("geog" "public"."geography", "max_segment_length" float8)
  RETURNS "public"."geography" AS '$libdir/postgis-3', 'geography_segmentize'
  LANGUAGE c IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_segmentize"("geog" "public"."geography", "max_segment_length" float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_seteffectivearea
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_seteffectivearea"("public"."geometry", float8, int4);
CREATE FUNCTION "public"."st_seteffectivearea"("public"."geometry", float8='-1'::integer, int4=1)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'LWGEOM_SetEffectiveArea'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_seteffectivearea"("public"."geometry", float8, int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_setpoint
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_setpoint"("public"."geometry", int4, "public"."geometry");
CREATE FUNCTION "public"."st_setpoint"("public"."geometry", int4, "public"."geometry")
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'LWGEOM_setpoint_linestring'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_setpoint"("public"."geometry", int4, "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_setsrid
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_setsrid"("geom" "public"."geometry", "srid" int4);
CREATE FUNCTION "public"."st_setsrid"("geom" "public"."geometry", "srid" int4)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'LWGEOM_set_srid'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."st_setsrid"("geom" "public"."geometry", "srid" int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_setsrid
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_setsrid"("geog" "public"."geography", "srid" int4);
CREATE FUNCTION "public"."st_setsrid"("geog" "public"."geography", "srid" int4)
  RETURNS "public"."geography" AS '$libdir/postgis-3', 'LWGEOM_set_srid'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_setsrid"("geog" "public"."geography", "srid" int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_sharedpaths
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_sharedpaths"("geom1" "public"."geometry", "geom2" "public"."geometry");
CREATE FUNCTION "public"."st_sharedpaths"("geom1" "public"."geometry", "geom2" "public"."geometry")
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'ST_SharedPaths'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_sharedpaths"("geom1" "public"."geometry", "geom2" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_shiftlongitude
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_shiftlongitude"("public"."geometry");
CREATE FUNCTION "public"."st_shiftlongitude"("public"."geometry")
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'LWGEOM_longitude_shift'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_shiftlongitude"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_shortestline
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_shortestline"(text, text);
CREATE FUNCTION "public"."st_shortestline"(text, text)
  RETURNS "public"."geometry" AS $BODY$ SELECT public.ST_ShortestLine($1::public.geometry, $2::public.geometry);  $BODY$
  LANGUAGE sql IMMUTABLE
  COST 100;
ALTER FUNCTION "public"."st_shortestline"(text, text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_shortestline
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_shortestline"("geom1" "public"."geometry", "geom2" "public"."geometry");
CREATE FUNCTION "public"."st_shortestline"("geom1" "public"."geometry", "geom2" "public"."geometry")
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'LWGEOM_shortestline2d'
  LANGUAGE c IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_shortestline"("geom1" "public"."geometry", "geom2" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_shortestline
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_shortestline"("public"."geography", "public"."geography", "use_spheroid" bool);
CREATE FUNCTION "public"."st_shortestline"("public"."geography", "public"."geography", "use_spheroid" bool=true)
  RETURNS "public"."geography" AS '$libdir/postgis-3', 'geography_shortestline'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."st_shortestline"("public"."geography", "public"."geography", "use_spheroid" bool) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_simplify
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_simplify"("public"."geometry", float8);
CREATE FUNCTION "public"."st_simplify"("public"."geometry", float8)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'LWGEOM_simplify2d'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_simplify"("public"."geometry", float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_simplify
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_simplify"("public"."geometry", float8, bool);
CREATE FUNCTION "public"."st_simplify"("public"."geometry", float8, bool)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'LWGEOM_simplify2d'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_simplify"("public"."geometry", float8, bool) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_simplifypolygonhull
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_simplifypolygonhull"("geom" "public"."geometry", "vertex_fraction" float8, "is_outer" bool);
CREATE FUNCTION "public"."st_simplifypolygonhull"("geom" "public"."geometry", "vertex_fraction" float8, "is_outer" bool=true)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'ST_SimplifyPolygonHull'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_simplifypolygonhull"("geom" "public"."geometry", "vertex_fraction" float8, "is_outer" bool) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_simplifypreservetopology
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_simplifypreservetopology"("public"."geometry", float8);
CREATE FUNCTION "public"."st_simplifypreservetopology"("public"."geometry", float8)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'topologypreservesimplify'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_simplifypreservetopology"("public"."geometry", float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_simplifyvw
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_simplifyvw"("public"."geometry", float8);
CREATE FUNCTION "public"."st_simplifyvw"("public"."geometry", float8)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'LWGEOM_SetEffectiveArea'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_simplifyvw"("public"."geometry", float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_snap
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_snap"("geom1" "public"."geometry", "geom2" "public"."geometry", float8);
CREATE FUNCTION "public"."st_snap"("geom1" "public"."geometry", "geom2" "public"."geometry", float8)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'ST_Snap'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_snap"("geom1" "public"."geometry", "geom2" "public"."geometry", float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_snaptogrid
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_snaptogrid"("public"."geometry", float8, float8, float8, float8);
CREATE FUNCTION "public"."st_snaptogrid"("public"."geometry", float8, float8, float8, float8)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'LWGEOM_snaptogrid'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_snaptogrid"("public"."geometry", float8, float8, float8, float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_snaptogrid
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_snaptogrid"("public"."geometry", float8, float8);
CREATE FUNCTION "public"."st_snaptogrid"("public"."geometry", float8, float8)
  RETURNS "public"."geometry" AS $BODY$SELECT public.ST_SnapToGrid($1, 0, 0, $2, $3)$BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_snaptogrid"("public"."geometry", float8, float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_snaptogrid
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_snaptogrid"("public"."geometry", float8);
CREATE FUNCTION "public"."st_snaptogrid"("public"."geometry", float8)
  RETURNS "public"."geometry" AS $BODY$SELECT public.ST_SnapToGrid($1, 0, 0, $2, $2)$BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_snaptogrid"("public"."geometry", float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_snaptogrid
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_snaptogrid"("geom1" "public"."geometry", "geom2" "public"."geometry", float8, float8, float8, float8);
CREATE FUNCTION "public"."st_snaptogrid"("geom1" "public"."geometry", "geom2" "public"."geometry", float8, float8, float8, float8)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'LWGEOM_snaptogrid_pointoff'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_snaptogrid"("geom1" "public"."geometry", "geom2" "public"."geometry", float8, float8, float8, float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_split
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_split"("geom1" "public"."geometry", "geom2" "public"."geometry");
CREATE FUNCTION "public"."st_split"("geom1" "public"."geometry", "geom2" "public"."geometry")
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'ST_Split'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_split"("geom1" "public"."geometry", "geom2" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_square
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_square"("size" float8, "cell_i" int4, "cell_j" int4, "origin" "public"."geometry");
CREATE FUNCTION "public"."st_square"("size" float8, "cell_i" int4, "cell_j" int4, "origin" "public"."geometry"='010100000000000000000000000000000000000000'::geometry)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'ST_Square'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_square"("size" float8, "cell_i" int4, "cell_j" int4, "origin" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_squaregrid
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_squaregrid"("size" float8, "bounds" "public"."geometry", OUT "geom" "public"."geometry", OUT "i" int4, OUT "j" int4);
CREATE FUNCTION "public"."st_squaregrid"(IN "size" float8, IN "bounds" "public"."geometry", OUT "geom" "public"."geometry", OUT "i" int4, OUT "j" int4)
  RETURNS SETOF "pg_catalog"."record" AS '$libdir/postgis-3', 'ST_ShapeGrid'
  LANGUAGE c IMMUTABLE STRICT
  COST 250
  ROWS 1000;
ALTER FUNCTION "public"."st_squaregrid"("size" float8, "bounds" "public"."geometry", OUT "geom" "public"."geometry", OUT "i" int4, OUT "j" int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_srid
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_srid"("geom" "public"."geometry");
CREATE FUNCTION "public"."st_srid"("geom" "public"."geometry")
  RETURNS "pg_catalog"."int4" AS '$libdir/postgis-3', 'LWGEOM_get_srid'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."st_srid"("geom" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_srid
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_srid"("geog" "public"."geography");
CREATE FUNCTION "public"."st_srid"("geog" "public"."geography")
  RETURNS "pg_catalog"."int4" AS '$libdir/postgis-3', 'LWGEOM_get_srid'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_srid"("geog" "public"."geography") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_startpoint
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_startpoint"("public"."geometry");
CREATE FUNCTION "public"."st_startpoint"("public"."geometry")
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'LWGEOM_startpoint_linestring'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_startpoint"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_subdivide
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_subdivide"("geom" "public"."geometry", "maxvertices" int4, "gridsize" float8);
CREATE FUNCTION "public"."st_subdivide"("geom" "public"."geometry", "maxvertices" int4=256, "gridsize" float8='-1.0'::numeric)
  RETURNS SETOF "public"."geometry" AS '$libdir/postgis-3', 'ST_Subdivide'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000
  ROWS 1000;
ALTER FUNCTION "public"."st_subdivide"("geom" "public"."geometry", "maxvertices" int4, "gridsize" float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_summary
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_summary"("public"."geography");
CREATE FUNCTION "public"."st_summary"("public"."geography")
  RETURNS "pg_catalog"."text" AS '$libdir/postgis-3', 'LWGEOM_summary'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_summary"("public"."geography") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_summary
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_summary"("public"."geometry");
CREATE FUNCTION "public"."st_summary"("public"."geometry")
  RETURNS "pg_catalog"."text" AS '$libdir/postgis-3', 'LWGEOM_summary'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_summary"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_swapordinates
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_swapordinates"("geom" "public"."geometry", "ords" cstring);
CREATE FUNCTION "public"."st_swapordinates"("geom" "public"."geometry", "ords" cstring)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'ST_SwapOrdinates'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_swapordinates"("geom" "public"."geometry", "ords" cstring) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_symdifference
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_symdifference"("geom1" "public"."geometry", "geom2" "public"."geometry", "gridsize" float8);
CREATE FUNCTION "public"."st_symdifference"("geom1" "public"."geometry", "geom2" "public"."geometry", "gridsize" float8='-1.0'::numeric)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'ST_SymDifference'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_symdifference"("geom1" "public"."geometry", "geom2" "public"."geometry", "gridsize" float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_symmetricdifference
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_symmetricdifference"("geom1" "public"."geometry", "geom2" "public"."geometry");
CREATE FUNCTION "public"."st_symmetricdifference"("geom1" "public"."geometry", "geom2" "public"."geometry")
  RETURNS "public"."geometry" AS $BODY$SELECT public.ST_SymDifference(geom1, geom2, -1.0);$BODY$
  LANGUAGE sql VOLATILE
  COST 100;
ALTER FUNCTION "public"."st_symmetricdifference"("geom1" "public"."geometry", "geom2" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_tileenvelope
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_tileenvelope"("zoom" int4, "x" int4, "y" int4, "bounds" "public"."geometry", "margin" float8);
CREATE FUNCTION "public"."st_tileenvelope"("zoom" int4, "x" int4, "y" int4, "bounds" "public"."geometry"='0102000020110F00000200000093107C45F81B73C193107C45F81B73C193107C45F81B734193107C45F81B7341'::geometry, "margin" float8=0.0)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'ST_TileEnvelope'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_tileenvelope"("zoom" int4, "x" int4, "y" int4, "bounds" "public"."geometry", "margin" float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_touches
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_touches"("geom1" "public"."geometry", "geom2" "public"."geometry");
CREATE FUNCTION "public"."st_touches"("geom1" "public"."geometry", "geom2" "public"."geometry")
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'touches'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_touches"("geom1" "public"."geometry", "geom2" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_transform
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_transform"("public"."geometry", int4);
CREATE FUNCTION "public"."st_transform"("public"."geometry", int4)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'transform'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_transform"("public"."geometry", int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_transform
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_transform"("geom" "public"."geometry", "from_proj" text, "to_srid" int4);
CREATE FUNCTION "public"."st_transform"("geom" "public"."geometry", "from_proj" text, "to_srid" int4)
  RETURNS "public"."geometry" AS $BODY$SELECT public.postgis_transform_geometry($1, $2, proj4text, $3)
	FROM public.spatial_ref_sys WHERE srid=$3;$BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_transform"("geom" "public"."geometry", "from_proj" text, "to_srid" int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_transform
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_transform"("geom" "public"."geometry", "from_proj" text, "to_proj" text);
CREATE FUNCTION "public"."st_transform"("geom" "public"."geometry", "from_proj" text, "to_proj" text)
  RETURNS "public"."geometry" AS $BODY$SELECT public.postgis_transform_geometry($1, $2, $3, 0)$BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_transform"("geom" "public"."geometry", "from_proj" text, "to_proj" text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_transform
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_transform"("geom" "public"."geometry", "to_proj" text);
CREATE FUNCTION "public"."st_transform"("geom" "public"."geometry", "to_proj" text)
  RETURNS "public"."geometry" AS $BODY$SELECT public.postgis_transform_geometry($1, proj4text, $2, 0)
	FROM public.spatial_ref_sys WHERE srid=public.ST_SRID($1);$BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_transform"("geom" "public"."geometry", "to_proj" text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_transformpipeline
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_transformpipeline"("geom" "public"."geometry", "pipeline" text, "to_srid" int4);
CREATE FUNCTION "public"."st_transformpipeline"("geom" "public"."geometry", "pipeline" text, "to_srid" int4=0)
  RETURNS "public"."geometry" AS $BODY$SELECT public.postgis_transform_pipeline_geometry($1, $2, TRUE, $3)$BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_transformpipeline"("geom" "public"."geometry", "pipeline" text, "to_srid" int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_translate
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_translate"("public"."geometry", float8, float8);
CREATE FUNCTION "public"."st_translate"("public"."geometry", float8, float8)
  RETURNS "public"."geometry" AS $BODY$SELECT public.ST_Translate($1, $2, $3, 0)$BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_translate"("public"."geometry", float8, float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_translate
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_translate"("public"."geometry", float8, float8, float8);
CREATE FUNCTION "public"."st_translate"("public"."geometry", float8, float8, float8)
  RETURNS "public"."geometry" AS $BODY$SELECT public.ST_Affine($1, 1, 0, 0, 0, 1, 0, 0, 0, 1, $2, $3, $4)$BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_translate"("public"."geometry", float8, float8, float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_transscale
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_transscale"("public"."geometry", float8, float8, float8, float8);
CREATE FUNCTION "public"."st_transscale"("public"."geometry", float8, float8, float8, float8)
  RETURNS "public"."geometry" AS $BODY$SELECT public.ST_Affine($1,  $4, 0, 0,  0, $5, 0,
		0, 0, 1,  $2 * $4, $3 * $5, 0)$BODY$
  LANGUAGE sql IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_transscale"("public"."geometry", float8, float8, float8, float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_triangulatepolygon
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_triangulatepolygon"("g1" "public"."geometry");
CREATE FUNCTION "public"."st_triangulatepolygon"("g1" "public"."geometry")
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'ST_TriangulatePolygon'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_triangulatepolygon"("g1" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_unaryunion
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_unaryunion"("public"."geometry", "gridsize" float8);
CREATE FUNCTION "public"."st_unaryunion"("public"."geometry", "gridsize" float8='-1.0'::numeric)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'ST_UnaryUnion'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_unaryunion"("public"."geometry", "gridsize" float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_union
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_union"("public"."_geometry");
CREATE FUNCTION "public"."st_union"("public"."_geometry")
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'pgis_union_geometry_array'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_union"("public"."_geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_union
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_union"("geom1" "public"."geometry", "geom2" "public"."geometry", "gridsize" float8);
CREATE FUNCTION "public"."st_union"("geom1" "public"."geometry", "geom2" "public"."geometry", "gridsize" float8)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'ST_Union'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_union"("geom1" "public"."geometry", "geom2" "public"."geometry", "gridsize" float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_union
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_union"("geom1" "public"."geometry", "geom2" "public"."geometry");
CREATE FUNCTION "public"."st_union"("geom1" "public"."geometry", "geom2" "public"."geometry")
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'ST_Union'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_union"("geom1" "public"."geometry", "geom2" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_voronoilines
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_voronoilines"("g1" "public"."geometry", "tolerance" float8, "extend_to" "public"."geometry");
CREATE FUNCTION "public"."st_voronoilines"("g1" "public"."geometry", "tolerance" float8=0.0, "extend_to" "public"."geometry"=NULL::geometry)
  RETURNS "public"."geometry" AS $BODY$ SELECT public._ST_Voronoi(g1, extend_to, tolerance, false) $BODY$
  LANGUAGE sql IMMUTABLE
  COST 100;
ALTER FUNCTION "public"."st_voronoilines"("g1" "public"."geometry", "tolerance" float8, "extend_to" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_voronoipolygons
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_voronoipolygons"("g1" "public"."geometry", "tolerance" float8, "extend_to" "public"."geometry");
CREATE FUNCTION "public"."st_voronoipolygons"("g1" "public"."geometry", "tolerance" float8=0.0, "extend_to" "public"."geometry"=NULL::geometry)
  RETURNS "public"."geometry" AS $BODY$ SELECT public._ST_Voronoi(g1, extend_to, tolerance, true) $BODY$
  LANGUAGE sql IMMUTABLE
  COST 100;
ALTER FUNCTION "public"."st_voronoipolygons"("g1" "public"."geometry", "tolerance" float8, "extend_to" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_within
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_within"("geom1" "public"."geometry", "geom2" "public"."geometry");
CREATE FUNCTION "public"."st_within"("geom1" "public"."geometry", "geom2" "public"."geometry")
  RETURNS "pg_catalog"."bool" AS '$libdir/postgis-3', 'within'
  LANGUAGE c IMMUTABLE STRICT
  COST 5000;
ALTER FUNCTION "public"."st_within"("geom1" "public"."geometry", "geom2" "public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_wkbtosql
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_wkbtosql"("wkb" bytea);
CREATE FUNCTION "public"."st_wkbtosql"("wkb" bytea)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'LWGEOM_from_WKB'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_wkbtosql"("wkb" bytea) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_wkttosql
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_wkttosql"(text);
CREATE FUNCTION "public"."st_wkttosql"(text)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'LWGEOM_from_text'
  LANGUAGE c IMMUTABLE STRICT
  COST 250;
ALTER FUNCTION "public"."st_wkttosql"(text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_wrapx
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_wrapx"("geom" "public"."geometry", "wrap" float8, "move" float8);
CREATE FUNCTION "public"."st_wrapx"("geom" "public"."geometry", "wrap" float8, "move" float8)
  RETURNS "public"."geometry" AS '$libdir/postgis-3', 'ST_WrapX'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."st_wrapx"("geom" "public"."geometry", "wrap" float8, "move" float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_x
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_x"("public"."geometry");
CREATE FUNCTION "public"."st_x"("public"."geometry")
  RETURNS "pg_catalog"."float8" AS '$libdir/postgis-3', 'LWGEOM_x_point'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."st_x"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_xmax
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_xmax"("public"."box3d");
CREATE FUNCTION "public"."st_xmax"("public"."box3d")
  RETURNS "pg_catalog"."float8" AS '$libdir/postgis-3', 'BOX3D_xmax'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."st_xmax"("public"."box3d") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_xmin
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_xmin"("public"."box3d");
CREATE FUNCTION "public"."st_xmin"("public"."box3d")
  RETURNS "pg_catalog"."float8" AS '$libdir/postgis-3', 'BOX3D_xmin'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."st_xmin"("public"."box3d") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_y
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_y"("public"."geometry");
CREATE FUNCTION "public"."st_y"("public"."geometry")
  RETURNS "pg_catalog"."float8" AS '$libdir/postgis-3', 'LWGEOM_y_point'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."st_y"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_ymax
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_ymax"("public"."box3d");
CREATE FUNCTION "public"."st_ymax"("public"."box3d")
  RETURNS "pg_catalog"."float8" AS '$libdir/postgis-3', 'BOX3D_ymax'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."st_ymax"("public"."box3d") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_ymin
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_ymin"("public"."box3d");
CREATE FUNCTION "public"."st_ymin"("public"."box3d")
  RETURNS "pg_catalog"."float8" AS '$libdir/postgis-3', 'BOX3D_ymin'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."st_ymin"("public"."box3d") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_z
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_z"("public"."geometry");
CREATE FUNCTION "public"."st_z"("public"."geometry")
  RETURNS "pg_catalog"."float8" AS '$libdir/postgis-3', 'LWGEOM_z_point'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."st_z"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_zmax
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_zmax"("public"."box3d");
CREATE FUNCTION "public"."st_zmax"("public"."box3d")
  RETURNS "pg_catalog"."float8" AS '$libdir/postgis-3', 'BOX3D_zmax'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."st_zmax"("public"."box3d") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_zmflag
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_zmflag"("public"."geometry");
CREATE FUNCTION "public"."st_zmflag"("public"."geometry")
  RETURNS "pg_catalog"."int2" AS '$libdir/postgis-3', 'LWGEOM_zmflag'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."st_zmflag"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for st_zmin
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."st_zmin"("public"."box3d");
CREATE FUNCTION "public"."st_zmin"("public"."box3d")
  RETURNS "pg_catalog"."float8" AS '$libdir/postgis-3', 'BOX3D_zmin'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."st_zmin"("public"."box3d") OWNER TO "postgres";

-- ----------------------------
-- Function structure for strict_word_similarity
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."strict_word_similarity"(text, text);
CREATE FUNCTION "public"."strict_word_similarity"(text, text)
  RETURNS "pg_catalog"."float4" AS '$libdir/pg_trgm', 'strict_word_similarity'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."strict_word_similarity"(text, text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for strict_word_similarity_commutator_op
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."strict_word_similarity_commutator_op"(text, text);
CREATE FUNCTION "public"."strict_word_similarity_commutator_op"(text, text)
  RETURNS "pg_catalog"."bool" AS '$libdir/pg_trgm', 'strict_word_similarity_commutator_op'
  LANGUAGE c STABLE STRICT
  COST 1;
ALTER FUNCTION "public"."strict_word_similarity_commutator_op"(text, text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for strict_word_similarity_dist_commutator_op
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."strict_word_similarity_dist_commutator_op"(text, text);
CREATE FUNCTION "public"."strict_word_similarity_dist_commutator_op"(text, text)
  RETURNS "pg_catalog"."float4" AS '$libdir/pg_trgm', 'strict_word_similarity_dist_commutator_op'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."strict_word_similarity_dist_commutator_op"(text, text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for strict_word_similarity_dist_op
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."strict_word_similarity_dist_op"(text, text);
CREATE FUNCTION "public"."strict_word_similarity_dist_op"(text, text)
  RETURNS "pg_catalog"."float4" AS '$libdir/pg_trgm', 'strict_word_similarity_dist_op'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."strict_word_similarity_dist_op"(text, text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for strict_word_similarity_op
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."strict_word_similarity_op"(text, text);
CREATE FUNCTION "public"."strict_word_similarity_op"(text, text)
  RETURNS "pg_catalog"."bool" AS '$libdir/pg_trgm', 'strict_word_similarity_op'
  LANGUAGE c STABLE STRICT
  COST 1;
ALTER FUNCTION "public"."strict_word_similarity_op"(text, text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for subvector
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."subvector"("public"."vector", int4, int4);
CREATE FUNCTION "public"."subvector"("public"."vector", int4, int4)
  RETURNS "public"."vector" AS '$libdir/vector', 'subvector'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."subvector"("public"."vector", int4, int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for subvector
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."subvector"("public"."halfvec", int4, int4);
CREATE FUNCTION "public"."subvector"("public"."halfvec", int4, int4)
  RETURNS "public"."halfvec" AS '$libdir/vector', 'halfvec_subvector'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."subvector"("public"."halfvec", int4, int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for text
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."text"("public"."geometry");
CREATE FUNCTION "public"."text"("public"."geometry")
  RETURNS "pg_catalog"."text" AS '$libdir/postgis-3', 'LWGEOM_to_text'
  LANGUAGE c IMMUTABLE STRICT
  COST 50;
ALTER FUNCTION "public"."text"("public"."geometry") OWNER TO "postgres";

-- ----------------------------
-- Function structure for text_soundex
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."text_soundex"(text);
CREATE FUNCTION "public"."text_soundex"(text)
  RETURNS "pg_catalog"."text" AS '$libdir/fuzzystrmatch', 'soundex'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."text_soundex"(text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for unify_prompt_placeholder
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."unify_prompt_placeholder"("input" text);
CREATE FUNCTION "public"."unify_prompt_placeholder"("input" text)
  RETURNS "pg_catalog"."text" AS $BODY$
DECLARE
    result TEXT := COALESCE(input, '');
    replacements TEXT[][] := ARRAY[
        -- Go template variables -> simple placeholders
        ['{{.Query}}', '{{query}}'],
        ['{{.Answer}}', '{{answer}}'],
        ['{{.CurrentTime}}', '{{current_time}}'],
        ['{{.CurrentWeek}}', '{{current_week}}'],
        ['{{.Yesterday}}', '{{yesterday}}'],
        ['{{.Contexts}}', '{{contexts}}'],
        -- Go template control structures -> simple placeholders or remove
        ['{{range .Contexts}}', '{{contexts}}'],
        -- Remove Go template syntax
        ['{{if .Contexts}}', ''],
        ['{{else}}', ''],
        ['{{.}}', '']
    ];
    r TEXT[];
BEGIN
    FOREACH r SLICE 1 IN ARRAY replacements LOOP
        result := REPLACE(result, r[1], r[2]);
    END LOOP;
    
    -- Handle {{range .Conversation}}...{{end}} block specially
    -- Replace the entire block with just {{conversation}}
    -- The pattern matches: {{range .Conversation}} followed by any content until {{end}}
    result := regexp_replace(
        result,
        '\{\{range \.Conversation\}\}[\s\S]*?\{\{end\}\}',
        '{{conversation}}',
        'g'
    );
    
    -- Clean up any remaining {{end}} tags
    result := REPLACE(result, '{{end}}', '');
    
    RETURN result;
END;
$BODY$
  LANGUAGE plpgsql VOLATILE
  COST 100;
ALTER FUNCTION "public"."unify_prompt_placeholder"("input" text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for update_mcp_services_updated_at
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."update_mcp_services_updated_at"();
CREATE FUNCTION "public"."update_mcp_services_updated_at"()
  RETURNS "pg_catalog"."trigger" AS $BODY$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$BODY$
  LANGUAGE plpgsql VOLATILE
  COST 100;
ALTER FUNCTION "public"."update_mcp_services_updated_at"() OWNER TO "postgres";

-- ----------------------------
-- Function structure for updategeometrysrid
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."updategeometrysrid"(varchar, varchar, varchar, int4);
CREATE FUNCTION "public"."updategeometrysrid"(varchar, varchar, varchar, int4)
  RETURNS "pg_catalog"."text" AS $BODY$
DECLARE
	ret  text;
BEGIN
	SELECT public.UpdateGeometrySRID('',$1,$2,$3,$4) into ret;
	RETURN ret;
END;
$BODY$
  LANGUAGE plpgsql VOLATILE STRICT
  COST 100;
ALTER FUNCTION "public"."updategeometrysrid"(varchar, varchar, varchar, int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for updategeometrysrid
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."updategeometrysrid"(varchar, varchar, int4);
CREATE FUNCTION "public"."updategeometrysrid"(varchar, varchar, int4)
  RETURNS "pg_catalog"."text" AS $BODY$
DECLARE
	ret  text;
BEGIN
	SELECT public.UpdateGeometrySRID('','',$1,$2,$3) into ret;
	RETURN ret;
END;
$BODY$
  LANGUAGE plpgsql VOLATILE STRICT
  COST 100;
ALTER FUNCTION "public"."updategeometrysrid"(varchar, varchar, int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for updategeometrysrid
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."updategeometrysrid"("catalogn_name" varchar, "schema_name" varchar, "table_name" varchar, "column_name" varchar, "new_srid_in" int4);
CREATE FUNCTION "public"."updategeometrysrid"("catalogn_name" varchar, "schema_name" varchar, "table_name" varchar, "column_name" varchar, "new_srid_in" int4)
  RETURNS "pg_catalog"."text" AS $BODY$
DECLARE
	myrec RECORD;
	okay boolean;
	cname varchar;
	real_schema name;
	unknown_srid integer;
	new_srid integer := new_srid_in;

BEGIN

	-- Find, check or fix schema_name
	IF ( schema_name != '' ) THEN
		okay = false;

		FOR myrec IN SELECT nspname FROM pg_namespace WHERE text(nspname) = schema_name LOOP
			okay := true;
		END LOOP;

		IF ( okay <> true ) THEN
			RAISE EXCEPTION 'Invalid schema name';
		ELSE
			real_schema = schema_name;
		END IF;
	ELSE
		SELECT INTO real_schema current_schema()::text;
	END IF;

	-- Ensure that column_name is in geometry_columns
	okay = false;
	FOR myrec IN SELECT type, coord_dimension FROM public.geometry_columns WHERE f_table_schema = text(real_schema) and f_table_name = table_name and f_geometry_column = column_name LOOP
		okay := true;
	END LOOP;
	IF (NOT okay) THEN
		RAISE EXCEPTION 'column not found in geometry_columns table';
		RETURN false;
	END IF;

	-- Ensure that new_srid is valid
	IF ( new_srid > 0 ) THEN
		IF ( SELECT count(*) = 0 from public.spatial_ref_sys where srid = new_srid ) THEN
			RAISE EXCEPTION 'invalid SRID: % not found in spatial_ref_sys', new_srid;
			RETURN false;
		END IF;
	ELSE
		unknown_srid := public.ST_SRID('POINT EMPTY'::public.geometry);
		IF ( new_srid != unknown_srid ) THEN
			new_srid := unknown_srid;
			RAISE NOTICE 'SRID value % converted to the officially unknown SRID value %', new_srid_in, new_srid;
		END IF;
	END IF;

	IF postgis_constraint_srid(real_schema, table_name, column_name) IS NOT NULL THEN
	-- srid was enforced with constraints before, keep it that way.
		-- Make up constraint name
		cname = 'enforce_srid_'  || column_name;

		-- Drop enforce_srid constraint
		EXECUTE 'ALTER TABLE ' || quote_ident(real_schema) ||
			'.' || quote_ident(table_name) ||
			' DROP constraint ' || quote_ident(cname);

		-- Update geometries SRID
		EXECUTE 'UPDATE ' || quote_ident(real_schema) ||
			'.' || quote_ident(table_name) ||
			' SET ' || quote_ident(column_name) ||
			' = public.ST_SetSRID(' || quote_ident(column_name) ||
			', ' || new_srid::text || ')';

		-- Reset enforce_srid constraint
		EXECUTE 'ALTER TABLE ' || quote_ident(real_schema) ||
			'.' || quote_ident(table_name) ||
			' ADD constraint ' || quote_ident(cname) ||
			' CHECK (st_srid(' || quote_ident(column_name) ||
			') = ' || new_srid::text || ')';
	ELSE
		-- We will use typmod to enforce if no srid constraints
		-- We are using postgis_type_name to lookup the new name
		-- (in case Paul changes his mind and flips geometry_columns to return old upper case name)
		EXECUTE 'ALTER TABLE ' || quote_ident(real_schema) || '.' || quote_ident(table_name) ||
		' ALTER COLUMN ' || quote_ident(column_name) || ' TYPE  geometry(' || public.postgis_type_name(myrec.type, myrec.coord_dimension, true) || ', ' || new_srid::text || ') USING public.ST_SetSRID(' || quote_ident(column_name) || ',' || new_srid::text || ');' ;
	END IF;

	RETURN real_schema || '.' || table_name || '.' || column_name ||' SRID changed to ' || new_srid::text;

END;
$BODY$
  LANGUAGE plpgsql VOLATILE STRICT
  COST 100;
ALTER FUNCTION "public"."updategeometrysrid"("catalogn_name" varchar, "schema_name" varchar, "table_name" varchar, "column_name" varchar, "new_srid_in" int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for uuid_generate_v1
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."uuid_generate_v1"();
CREATE FUNCTION "public"."uuid_generate_v1"()
  RETURNS "pg_catalog"."uuid" AS '$libdir/uuid-ossp', 'uuid_generate_v1'
  LANGUAGE c VOLATILE STRICT
  COST 1;
ALTER FUNCTION "public"."uuid_generate_v1"() OWNER TO "postgres";

-- ----------------------------
-- Function structure for uuid_generate_v1mc
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."uuid_generate_v1mc"();
CREATE FUNCTION "public"."uuid_generate_v1mc"()
  RETURNS "pg_catalog"."uuid" AS '$libdir/uuid-ossp', 'uuid_generate_v1mc'
  LANGUAGE c VOLATILE STRICT
  COST 1;
ALTER FUNCTION "public"."uuid_generate_v1mc"() OWNER TO "postgres";

-- ----------------------------
-- Function structure for uuid_generate_v3
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."uuid_generate_v3"("namespace" uuid, "name" text);
CREATE FUNCTION "public"."uuid_generate_v3"("namespace" uuid, "name" text)
  RETURNS "pg_catalog"."uuid" AS '$libdir/uuid-ossp', 'uuid_generate_v3'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."uuid_generate_v3"("namespace" uuid, "name" text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for uuid_generate_v4
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."uuid_generate_v4"();
CREATE FUNCTION "public"."uuid_generate_v4"()
  RETURNS "pg_catalog"."uuid" AS '$libdir/uuid-ossp', 'uuid_generate_v4'
  LANGUAGE c VOLATILE STRICT
  COST 1;
ALTER FUNCTION "public"."uuid_generate_v4"() OWNER TO "postgres";

-- ----------------------------
-- Function structure for uuid_generate_v5
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."uuid_generate_v5"("namespace" uuid, "name" text);
CREATE FUNCTION "public"."uuid_generate_v5"("namespace" uuid, "name" text)
  RETURNS "pg_catalog"."uuid" AS '$libdir/uuid-ossp', 'uuid_generate_v5'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."uuid_generate_v5"("namespace" uuid, "name" text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for uuid_nil
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."uuid_nil"();
CREATE FUNCTION "public"."uuid_nil"()
  RETURNS "pg_catalog"."uuid" AS '$libdir/uuid-ossp', 'uuid_nil'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."uuid_nil"() OWNER TO "postgres";

-- ----------------------------
-- Function structure for uuid_ns_dns
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."uuid_ns_dns"();
CREATE FUNCTION "public"."uuid_ns_dns"()
  RETURNS "pg_catalog"."uuid" AS '$libdir/uuid-ossp', 'uuid_ns_dns'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."uuid_ns_dns"() OWNER TO "postgres";

-- ----------------------------
-- Function structure for uuid_ns_oid
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."uuid_ns_oid"();
CREATE FUNCTION "public"."uuid_ns_oid"()
  RETURNS "pg_catalog"."uuid" AS '$libdir/uuid-ossp', 'uuid_ns_oid'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."uuid_ns_oid"() OWNER TO "postgres";

-- ----------------------------
-- Function structure for uuid_ns_url
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."uuid_ns_url"();
CREATE FUNCTION "public"."uuid_ns_url"()
  RETURNS "pg_catalog"."uuid" AS '$libdir/uuid-ossp', 'uuid_ns_url'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."uuid_ns_url"() OWNER TO "postgres";

-- ----------------------------
-- Function structure for uuid_ns_x500
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."uuid_ns_x500"();
CREATE FUNCTION "public"."uuid_ns_x500"()
  RETURNS "pg_catalog"."uuid" AS '$libdir/uuid-ossp', 'uuid_ns_x500'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."uuid_ns_x500"() OWNER TO "postgres";

-- ----------------------------
-- Function structure for vector
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."vector"("public"."vector", int4, bool);
CREATE FUNCTION "public"."vector"("public"."vector", int4, bool)
  RETURNS "public"."vector" AS '$libdir/vector', 'vector'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."vector"("public"."vector", int4, bool) OWNER TO "postgres";

-- ----------------------------
-- Function structure for vector_accum
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."vector_accum"(_float8, "public"."vector");
CREATE FUNCTION "public"."vector_accum"(_float8, "public"."vector")
  RETURNS "pg_catalog"."_float8" AS '$libdir/vector', 'vector_accum'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."vector_accum"(_float8, "public"."vector") OWNER TO "postgres";

-- ----------------------------
-- Function structure for vector_add
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."vector_add"("public"."vector", "public"."vector");
CREATE FUNCTION "public"."vector_add"("public"."vector", "public"."vector")
  RETURNS "public"."vector" AS '$libdir/vector', 'vector_add'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."vector_add"("public"."vector", "public"."vector") OWNER TO "postgres";

-- ----------------------------
-- Function structure for vector_avg
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."vector_avg"(_float8);
CREATE FUNCTION "public"."vector_avg"(_float8)
  RETURNS "public"."vector" AS '$libdir/vector', 'vector_avg'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."vector_avg"(_float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for vector_cmp
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."vector_cmp"("public"."vector", "public"."vector");
CREATE FUNCTION "public"."vector_cmp"("public"."vector", "public"."vector")
  RETURNS "pg_catalog"."int4" AS '$libdir/vector', 'vector_cmp'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."vector_cmp"("public"."vector", "public"."vector") OWNER TO "postgres";

-- ----------------------------
-- Function structure for vector_combine
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."vector_combine"(_float8, _float8);
CREATE FUNCTION "public"."vector_combine"(_float8, _float8)
  RETURNS "pg_catalog"."_float8" AS '$libdir/vector', 'vector_combine'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."vector_combine"(_float8, _float8) OWNER TO "postgres";

-- ----------------------------
-- Function structure for vector_concat
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."vector_concat"("public"."vector", "public"."vector");
CREATE FUNCTION "public"."vector_concat"("public"."vector", "public"."vector")
  RETURNS "public"."vector" AS '$libdir/vector', 'vector_concat'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."vector_concat"("public"."vector", "public"."vector") OWNER TO "postgres";

-- ----------------------------
-- Function structure for vector_dims
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."vector_dims"("public"."halfvec");
CREATE FUNCTION "public"."vector_dims"("public"."halfvec")
  RETURNS "pg_catalog"."int4" AS '$libdir/vector', 'halfvec_vector_dims'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."vector_dims"("public"."halfvec") OWNER TO "postgres";

-- ----------------------------
-- Function structure for vector_dims
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."vector_dims"("public"."vector");
CREATE FUNCTION "public"."vector_dims"("public"."vector")
  RETURNS "pg_catalog"."int4" AS '$libdir/vector', 'vector_dims'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."vector_dims"("public"."vector") OWNER TO "postgres";

-- ----------------------------
-- Function structure for vector_eq
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."vector_eq"("public"."vector", "public"."vector");
CREATE FUNCTION "public"."vector_eq"("public"."vector", "public"."vector")
  RETURNS "pg_catalog"."bool" AS '$libdir/vector', 'vector_eq'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."vector_eq"("public"."vector", "public"."vector") OWNER TO "postgres";

-- ----------------------------
-- Function structure for vector_ge
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."vector_ge"("public"."vector", "public"."vector");
CREATE FUNCTION "public"."vector_ge"("public"."vector", "public"."vector")
  RETURNS "pg_catalog"."bool" AS '$libdir/vector', 'vector_ge'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."vector_ge"("public"."vector", "public"."vector") OWNER TO "postgres";

-- ----------------------------
-- Function structure for vector_gt
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."vector_gt"("public"."vector", "public"."vector");
CREATE FUNCTION "public"."vector_gt"("public"."vector", "public"."vector")
  RETURNS "pg_catalog"."bool" AS '$libdir/vector', 'vector_gt'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."vector_gt"("public"."vector", "public"."vector") OWNER TO "postgres";

-- ----------------------------
-- Function structure for vector_in
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."vector_in"(cstring, oid, int4);
CREATE FUNCTION "public"."vector_in"(cstring, oid, int4)
  RETURNS "public"."vector" AS '$libdir/vector', 'vector_in'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."vector_in"(cstring, oid, int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for vector_l2_squared_distance
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."vector_l2_squared_distance"("public"."vector", "public"."vector");
CREATE FUNCTION "public"."vector_l2_squared_distance"("public"."vector", "public"."vector")
  RETURNS "pg_catalog"."float8" AS '$libdir/vector', 'vector_l2_squared_distance'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."vector_l2_squared_distance"("public"."vector", "public"."vector") OWNER TO "postgres";

-- ----------------------------
-- Function structure for vector_le
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."vector_le"("public"."vector", "public"."vector");
CREATE FUNCTION "public"."vector_le"("public"."vector", "public"."vector")
  RETURNS "pg_catalog"."bool" AS '$libdir/vector', 'vector_le'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."vector_le"("public"."vector", "public"."vector") OWNER TO "postgres";

-- ----------------------------
-- Function structure for vector_lt
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."vector_lt"("public"."vector", "public"."vector");
CREATE FUNCTION "public"."vector_lt"("public"."vector", "public"."vector")
  RETURNS "pg_catalog"."bool" AS '$libdir/vector', 'vector_lt'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."vector_lt"("public"."vector", "public"."vector") OWNER TO "postgres";

-- ----------------------------
-- Function structure for vector_mul
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."vector_mul"("public"."vector", "public"."vector");
CREATE FUNCTION "public"."vector_mul"("public"."vector", "public"."vector")
  RETURNS "public"."vector" AS '$libdir/vector', 'vector_mul'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."vector_mul"("public"."vector", "public"."vector") OWNER TO "postgres";

-- ----------------------------
-- Function structure for vector_ne
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."vector_ne"("public"."vector", "public"."vector");
CREATE FUNCTION "public"."vector_ne"("public"."vector", "public"."vector")
  RETURNS "pg_catalog"."bool" AS '$libdir/vector', 'vector_ne'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."vector_ne"("public"."vector", "public"."vector") OWNER TO "postgres";

-- ----------------------------
-- Function structure for vector_negative_inner_product
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."vector_negative_inner_product"("public"."vector", "public"."vector");
CREATE FUNCTION "public"."vector_negative_inner_product"("public"."vector", "public"."vector")
  RETURNS "pg_catalog"."float8" AS '$libdir/vector', 'vector_negative_inner_product'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."vector_negative_inner_product"("public"."vector", "public"."vector") OWNER TO "postgres";

-- ----------------------------
-- Function structure for vector_norm
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."vector_norm"("public"."vector");
CREATE FUNCTION "public"."vector_norm"("public"."vector")
  RETURNS "pg_catalog"."float8" AS '$libdir/vector', 'vector_norm'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."vector_norm"("public"."vector") OWNER TO "postgres";

-- ----------------------------
-- Function structure for vector_out
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."vector_out"("public"."vector");
CREATE FUNCTION "public"."vector_out"("public"."vector")
  RETURNS "pg_catalog"."cstring" AS '$libdir/vector', 'vector_out'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."vector_out"("public"."vector") OWNER TO "postgres";

-- ----------------------------
-- Function structure for vector_recv
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."vector_recv"(internal, oid, int4);
CREATE FUNCTION "public"."vector_recv"(internal, oid, int4)
  RETURNS "public"."vector" AS '$libdir/vector', 'vector_recv'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."vector_recv"(internal, oid, int4) OWNER TO "postgres";

-- ----------------------------
-- Function structure for vector_send
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."vector_send"("public"."vector");
CREATE FUNCTION "public"."vector_send"("public"."vector")
  RETURNS "pg_catalog"."bytea" AS '$libdir/vector', 'vector_send'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."vector_send"("public"."vector") OWNER TO "postgres";

-- ----------------------------
-- Function structure for vector_spherical_distance
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."vector_spherical_distance"("public"."vector", "public"."vector");
CREATE FUNCTION "public"."vector_spherical_distance"("public"."vector", "public"."vector")
  RETURNS "pg_catalog"."float8" AS '$libdir/vector', 'vector_spherical_distance'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."vector_spherical_distance"("public"."vector", "public"."vector") OWNER TO "postgres";

-- ----------------------------
-- Function structure for vector_sub
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."vector_sub"("public"."vector", "public"."vector");
CREATE FUNCTION "public"."vector_sub"("public"."vector", "public"."vector")
  RETURNS "public"."vector" AS '$libdir/vector', 'vector_sub'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."vector_sub"("public"."vector", "public"."vector") OWNER TO "postgres";

-- ----------------------------
-- Function structure for vector_to_float4
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."vector_to_float4"("public"."vector", int4, bool);
CREATE FUNCTION "public"."vector_to_float4"("public"."vector", int4, bool)
  RETURNS "pg_catalog"."_float4" AS '$libdir/vector', 'vector_to_float4'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."vector_to_float4"("public"."vector", int4, bool) OWNER TO "postgres";

-- ----------------------------
-- Function structure for vector_to_halfvec
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."vector_to_halfvec"("public"."vector", int4, bool);
CREATE FUNCTION "public"."vector_to_halfvec"("public"."vector", int4, bool)
  RETURNS "public"."halfvec" AS '$libdir/vector', 'vector_to_halfvec'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."vector_to_halfvec"("public"."vector", int4, bool) OWNER TO "postgres";

-- ----------------------------
-- Function structure for vector_to_sparsevec
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."vector_to_sparsevec"("public"."vector", int4, bool);
CREATE FUNCTION "public"."vector_to_sparsevec"("public"."vector", int4, bool)
  RETURNS "public"."sparsevec" AS '$libdir/vector', 'vector_to_sparsevec'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."vector_to_sparsevec"("public"."vector", int4, bool) OWNER TO "postgres";

-- ----------------------------
-- Function structure for vector_typmod_in
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."vector_typmod_in"(_cstring);
CREATE FUNCTION "public"."vector_typmod_in"(_cstring)
  RETURNS "pg_catalog"."int4" AS '$libdir/vector', 'vector_typmod_in'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."vector_typmod_in"(_cstring) OWNER TO "postgres";

-- ----------------------------
-- Function structure for word_similarity
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."word_similarity"(text, text);
CREATE FUNCTION "public"."word_similarity"(text, text)
  RETURNS "pg_catalog"."float4" AS '$libdir/pg_trgm', 'word_similarity'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."word_similarity"(text, text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for word_similarity_commutator_op
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."word_similarity_commutator_op"(text, text);
CREATE FUNCTION "public"."word_similarity_commutator_op"(text, text)
  RETURNS "pg_catalog"."bool" AS '$libdir/pg_trgm', 'word_similarity_commutator_op'
  LANGUAGE c STABLE STRICT
  COST 1;
ALTER FUNCTION "public"."word_similarity_commutator_op"(text, text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for word_similarity_dist_commutator_op
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."word_similarity_dist_commutator_op"(text, text);
CREATE FUNCTION "public"."word_similarity_dist_commutator_op"(text, text)
  RETURNS "pg_catalog"."float4" AS '$libdir/pg_trgm', 'word_similarity_dist_commutator_op'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."word_similarity_dist_commutator_op"(text, text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for word_similarity_dist_op
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."word_similarity_dist_op"(text, text);
CREATE FUNCTION "public"."word_similarity_dist_op"(text, text)
  RETURNS "pg_catalog"."float4" AS '$libdir/pg_trgm', 'word_similarity_dist_op'
  LANGUAGE c IMMUTABLE STRICT
  COST 1;
ALTER FUNCTION "public"."word_similarity_dist_op"(text, text) OWNER TO "postgres";

-- ----------------------------
-- Function structure for word_similarity_op
-- ----------------------------
DROP FUNCTION IF EXISTS "public"."word_similarity_op"(text, text);
CREATE FUNCTION "public"."word_similarity_op"(text, text)
  RETURNS "pg_catalog"."bool" AS '$libdir/pg_trgm', 'word_similarity_op'
  LANGUAGE c STABLE STRICT
  COST 1;
ALTER FUNCTION "public"."word_similarity_op"(text, text) OWNER TO "postgres";

-- ----------------------------
-- View structure for geography_columns
-- ----------------------------
DROP VIEW IF EXISTS "public"."geography_columns";
CREATE VIEW "public"."geography_columns" AS  SELECT current_database() AS f_table_catalog,
    n.nspname AS f_table_schema,
    c.relname AS f_table_name,
    a.attname AS f_geography_column,
    postgis_typmod_dims(a.atttypmod) AS coord_dimension,
    postgis_typmod_srid(a.atttypmod) AS srid,
    postgis_typmod_type(a.atttypmod) AS type
   FROM pg_class c,
    pg_attribute a,
    pg_type t,
    pg_namespace n
  WHERE t.typname = 'geography'::name AND a.attisdropped = false AND a.atttypid = t.oid AND a.attrelid = c.oid AND c.relnamespace = n.oid AND (c.relkind = ANY (ARRAY['r'::"char", 'v'::"char", 'm'::"char", 'f'::"char", 'p'::"char"])) AND NOT pg_is_other_temp_schema(c.relnamespace) AND has_table_privilege(c.oid, 'SELECT'::text);
ALTER TABLE "public"."geography_columns" OWNER TO "postgres";

-- ----------------------------
-- View structure for geometry_columns
-- ----------------------------
DROP VIEW IF EXISTS "public"."geometry_columns";
CREATE VIEW "public"."geometry_columns" AS  SELECT current_database()::character varying(256) AS f_table_catalog,
    n.nspname AS f_table_schema,
    c.relname AS f_table_name,
    a.attname AS f_geometry_column,
    COALESCE(postgis_typmod_dims(a.atttypmod), 2) AS coord_dimension,
    COALESCE(NULLIF(postgis_typmod_srid(a.atttypmod), 0), 0) AS srid,
    replace(replace(COALESCE(NULLIF(upper(postgis_typmod_type(a.atttypmod)), 'GEOMETRY'::text), 'GEOMETRY'::text), 'ZM'::text, ''::text), 'Z'::text, ''::text)::character varying(30) AS type
   FROM pg_class c
     JOIN pg_attribute a ON a.attrelid = c.oid AND NOT a.attisdropped
     JOIN pg_namespace n ON c.relnamespace = n.oid
     JOIN pg_type t ON a.atttypid = t.oid
  WHERE (c.relkind = ANY (ARRAY['r'::"char", 'v'::"char", 'm'::"char", 'f'::"char", 'p'::"char"])) AND NOT c.relname = 'raster_columns'::name AND t.typname = 'geometry'::name AND NOT pg_is_other_temp_schema(c.relnamespace) AND has_table_privilege(c.oid, 'SELECT'::text);
ALTER TABLE "public"."geometry_columns" OWNER TO "postgres";
CREATE RULE "geometry_columns_insert" AS ON INSERT TO "public"."geometry_columns" DO INSTEAD NOTHING;;
CREATE RULE "geometry_columns_update" AS ON UPDATE TO "public"."geometry_columns" DO INSTEAD NOTHING;;
CREATE RULE "geometry_columns_delete" AS ON DELETE TO "public"."geometry_columns" DO INSTEAD NOTHING;;

-- ----------------------------
-- Alter sequences owned by
-- ----------------------------
ALTER SEQUENCE "public"."embeddings_id_seq"
OWNED BY "public"."embeddings"."id";
SELECT setval('"public"."embeddings_id_seq"', 1, false);

-- ----------------------------
-- Alter sequences owned by
-- ----------------------------
ALTER SEQUENCE "public"."tenants_id_seq"
OWNED BY "public"."tenants"."id";
SELECT setval('"public"."tenants_id_seq"', 10000, true);

-- ----------------------------
-- Indexes structure for table auth_tokens
-- ----------------------------
CREATE INDEX "idx_auth_tokens_expires_at" ON "public"."auth_tokens" USING btree (
  "expires_at" "pg_catalog"."timestamptz_ops" ASC NULLS LAST
);
CREATE INDEX "idx_auth_tokens_token" ON "public"."auth_tokens" USING btree (
  "token" COLLATE "pg_catalog"."default" "pg_catalog"."text_ops" ASC NULLS LAST
);
CREATE INDEX "idx_auth_tokens_token_type" ON "public"."auth_tokens" USING btree (
  "token_type" COLLATE "pg_catalog"."default" "pg_catalog"."text_ops" ASC NULLS LAST
);
CREATE INDEX "idx_auth_tokens_user_id" ON "public"."auth_tokens" USING btree (
  "user_id" COLLATE "pg_catalog"."default" "pg_catalog"."text_ops" ASC NULLS LAST
);

-- ----------------------------
-- Primary Key structure for table auth_tokens
-- ----------------------------
ALTER TABLE "public"."auth_tokens" ADD CONSTRAINT "auth_tokens_pkey" PRIMARY KEY ("id");

-- ----------------------------
-- Indexes structure for table chunks
-- ----------------------------
CREATE INDEX "idx_chunks_chunk_type" ON "public"."chunks" USING btree (
  "chunk_type" COLLATE "pg_catalog"."default" "pg_catalog"."text_ops" ASC NULLS LAST
);
CREATE INDEX "idx_chunks_content_hash" ON "public"."chunks" USING btree (
  "content_hash" COLLATE "pg_catalog"."default" "pg_catalog"."text_ops" ASC NULLS LAST
);
CREATE INDEX "idx_chunks_parent_id" ON "public"."chunks" USING btree (
  "parent_chunk_id" COLLATE "pg_catalog"."default" "pg_catalog"."text_ops" ASC NULLS LAST
);
CREATE INDEX "idx_chunks_tag" ON "public"."chunks" USING btree (
  "tag_id" COLLATE "pg_catalog"."default" "pg_catalog"."text_ops" ASC NULLS LAST
);
CREATE INDEX "idx_chunks_tenant_kg" ON "public"."chunks" USING btree (
  "tenant_id" "pg_catalog"."int4_ops" ASC NULLS LAST,
  "knowledge_id" COLLATE "pg_catalog"."default" "pg_catalog"."text_ops" ASC NULLS LAST
);

-- ----------------------------
-- Primary Key structure for table chunks
-- ----------------------------
ALTER TABLE "public"."chunks" ADD CONSTRAINT "chunks_pkey" PRIMARY KEY ("id");

-- ----------------------------
-- Indexes structure for table custom_agents
-- ----------------------------
CREATE INDEX "idx_custom_agents_deleted_at" ON "public"."custom_agents" USING btree (
  "deleted_at" "pg_catalog"."timestamptz_ops" ASC NULLS LAST
);
CREATE INDEX "idx_custom_agents_is_builtin" ON "public"."custom_agents" USING btree (
  "is_builtin" "pg_catalog"."bool_ops" ASC NULLS LAST
);
CREATE INDEX "idx_custom_agents_tenant_id" ON "public"."custom_agents" USING btree (
  "tenant_id" "pg_catalog"."int4_ops" ASC NULLS LAST
);

-- ----------------------------
-- Primary Key structure for table custom_agents
-- ----------------------------
ALTER TABLE "public"."custom_agents" ADD CONSTRAINT "custom_agents_pkey" PRIMARY KEY ("id", "tenant_id");

-- ----------------------------
-- Indexes structure for table embeddings
-- ----------------------------
CREATE INDEX "embeddings_embedding_idx_3584" ON "public"."embeddings" USING hnsw (
  (embedding::halfvec(3584)) "public"."halfvec_cosine_ops"
) WHERE dimension = 3584;
CREATE INDEX "embeddings_embedding_idx_798" ON "public"."embeddings" USING hnsw (
  (embedding::halfvec(798)) "public"."halfvec_cosine_ops"
) WHERE dimension = 798;
CREATE INDEX "embeddings_search_idx" ON "public"."embeddings" (
  "id" "paradedb"."anyelement_bm25_ops" ASC NULLS LAST,
  "knowledge_base_id" COLLATE "pg_catalog"."default" "paradedb"."anyelement_bm25_ops" ASC NULLS LAST,
  "content" COLLATE "pg_catalog"."default" "paradedb"."anyelement_bm25_ops" ASC NULLS LAST,
  "knowledge_id" COLLATE "pg_catalog"."default" "paradedb"."anyelement_bm25_ops" ASC NULLS LAST,
  "chunk_id" COLLATE "pg_catalog"."default" "paradedb"."anyelement_bm25_ops" ASC NULLS LAST
);
CREATE UNIQUE INDEX "embeddings_unique_source" ON "public"."embeddings" USING btree (
  "source_id" COLLATE "pg_catalog"."default" "pg_catalog"."text_ops" ASC NULLS LAST,
  "source_type" "pg_catalog"."int4_ops" ASC NULLS LAST
);
CREATE INDEX "idx_embeddings_is_enabled" ON "public"."embeddings" USING btree (
  "is_enabled" "pg_catalog"."bool_ops" ASC NULLS LAST
);
CREATE INDEX "idx_embeddings_knowledge_base_id" ON "public"."embeddings" USING btree (
  "knowledge_base_id" COLLATE "pg_catalog"."default" "pg_catalog"."text_ops" ASC NULLS LAST
);
CREATE INDEX "idx_embeddings_tag_id" ON "public"."embeddings" USING btree (
  "tag_id" COLLATE "pg_catalog"."default" "pg_catalog"."text_ops" ASC NULLS LAST
);

-- ----------------------------
-- Primary Key structure for table embeddings
-- ----------------------------
ALTER TABLE "public"."embeddings" ADD CONSTRAINT "embeddings_pkey" PRIMARY KEY ("id");

-- ----------------------------
-- Indexes structure for table knowledge_bases
-- ----------------------------
CREATE INDEX "idx_knowledge_bases_tenant_id" ON "public"."knowledge_bases" USING btree (
  "tenant_id" "pg_catalog"."int4_ops" ASC NULLS LAST
);

-- ----------------------------
-- Primary Key structure for table knowledge_bases
-- ----------------------------
ALTER TABLE "public"."knowledge_bases" ADD CONSTRAINT "knowledge_bases_pkey" PRIMARY KEY ("id");

-- ----------------------------
-- Indexes structure for table knowledge_tags
-- ----------------------------
CREATE INDEX "idx_knowledge_tags_kb" ON "public"."knowledge_tags" USING btree (
  "tenant_id" "pg_catalog"."int4_ops" ASC NULLS LAST,
  "knowledge_base_id" COLLATE "pg_catalog"."default" "pg_catalog"."text_ops" ASC NULLS LAST
);
CREATE UNIQUE INDEX "idx_knowledge_tags_kb_name" ON "public"."knowledge_tags" USING btree (
  "tenant_id" "pg_catalog"."int4_ops" ASC NULLS LAST,
  "knowledge_base_id" COLLATE "pg_catalog"."default" "pg_catalog"."text_ops" ASC NULLS LAST,
  "name" COLLATE "pg_catalog"."default" "pg_catalog"."text_ops" ASC NULLS LAST
);

-- ----------------------------
-- Primary Key structure for table knowledge_tags
-- ----------------------------
ALTER TABLE "public"."knowledge_tags" ADD CONSTRAINT "knowledge_tags_pkey" PRIMARY KEY ("id");

-- ----------------------------
-- Indexes structure for table knowledges
-- ----------------------------
CREATE INDEX "idx_knowledges_base_id" ON "public"."knowledges" USING btree (
  "knowledge_base_id" COLLATE "pg_catalog"."default" "pg_catalog"."text_ops" ASC NULLS LAST
);
CREATE INDEX "idx_knowledges_enable_status" ON "public"."knowledges" USING btree (
  "enable_status" COLLATE "pg_catalog"."default" "pg_catalog"."text_ops" ASC NULLS LAST
);
CREATE INDEX "idx_knowledges_parse_status" ON "public"."knowledges" USING btree (
  "parse_status" COLLATE "pg_catalog"."default" "pg_catalog"."text_ops" ASC NULLS LAST
);
CREATE INDEX "idx_knowledges_summary_status" ON "public"."knowledges" USING btree (
  "summary_status" COLLATE "pg_catalog"."default" "pg_catalog"."text_ops" ASC NULLS LAST
);
CREATE INDEX "idx_knowledges_tag" ON "public"."knowledges" USING btree (
  "tag_id" COLLATE "pg_catalog"."default" "pg_catalog"."text_ops" ASC NULLS LAST
);
CREATE INDEX "idx_knowledges_tenant_id" ON "public"."knowledges" USING btree (
  "tenant_id" "pg_catalog"."int4_ops" ASC NULLS LAST
);

-- ----------------------------
-- Primary Key structure for table knowledges
-- ----------------------------
ALTER TABLE "public"."knowledges" ADD CONSTRAINT "knowledges_pkey" PRIMARY KEY ("id");

-- ----------------------------
-- Indexes structure for table mcp_services
-- ----------------------------
CREATE INDEX "idx_mcp_services_deleted_at" ON "public"."mcp_services" USING btree (
  "deleted_at" "pg_catalog"."timestamp_ops" ASC NULLS LAST
);
CREATE INDEX "idx_mcp_services_enabled" ON "public"."mcp_services" USING btree (
  "enabled" "pg_catalog"."bool_ops" ASC NULLS LAST
);
CREATE INDEX "idx_mcp_services_tenant_id" ON "public"."mcp_services" USING btree (
  "tenant_id" "pg_catalog"."int4_ops" ASC NULLS LAST
);

-- ----------------------------
-- Triggers structure for table mcp_services
-- ----------------------------
CREATE TRIGGER "trigger_mcp_services_updated_at" BEFORE UPDATE ON "public"."mcp_services"
FOR EACH ROW
EXECUTE PROCEDURE "public"."update_mcp_services_updated_at"();

-- ----------------------------
-- Primary Key structure for table mcp_services
-- ----------------------------
ALTER TABLE "public"."mcp_services" ADD CONSTRAINT "mcp_services_pkey" PRIMARY KEY ("id");

-- ----------------------------
-- Indexes structure for table messages
-- ----------------------------
CREATE INDEX "idx_messages_agent_steps" ON "public"."messages" USING gin (
  "agent_steps" "pg_catalog"."jsonb_ops"
);
CREATE INDEX "idx_messages_session_id" ON "public"."messages" USING btree (
  "session_id" COLLATE "pg_catalog"."default" "pg_catalog"."text_ops" ASC NULLS LAST
);

-- ----------------------------
-- Primary Key structure for table messages
-- ----------------------------
ALTER TABLE "public"."messages" ADD CONSTRAINT "messages_pkey" PRIMARY KEY ("id");

-- ----------------------------
-- Indexes structure for table models
-- ----------------------------
CREATE INDEX "idx_models_is_builtin" ON "public"."models" USING btree (
  "is_builtin" "pg_catalog"."bool_ops" ASC NULLS LAST
);
CREATE INDEX "idx_models_source" ON "public"."models" USING btree (
  "source" COLLATE "pg_catalog"."default" "pg_catalog"."text_ops" ASC NULLS LAST
);
CREATE INDEX "idx_models_type" ON "public"."models" USING btree (
  "type" COLLATE "pg_catalog"."default" "pg_catalog"."text_ops" ASC NULLS LAST
);

-- ----------------------------
-- Primary Key structure for table models
-- ----------------------------
ALTER TABLE "public"."models" ADD CONSTRAINT "models_pkey" PRIMARY KEY ("id");

-- ----------------------------
-- Primary Key structure for table schema_migrations
-- ----------------------------
ALTER TABLE "public"."schema_migrations" ADD CONSTRAINT "schema_migrations_pkey" PRIMARY KEY ("version");

-- ----------------------------
-- Indexes structure for table session_items
-- ----------------------------
CREATE INDEX "idx_session_items_message_id" ON "public"."session_items" USING btree (
  "message_id" COLLATE "pg_catalog"."default" "pg_catalog"."text_ops" ASC NULLS LAST
);
CREATE INDEX "idx_session_items_session_id" ON "public"."session_items" USING btree (
  "session_id" COLLATE "pg_catalog"."default" "pg_catalog"."text_ops" ASC NULLS LAST
);
CREATE INDEX "idx_session_items_session_sort" ON "public"."session_items" USING btree (
  "session_id" COLLATE "pg_catalog"."default" "pg_catalog"."text_ops" ASC NULLS LAST,
  "sort_order" "pg_catalog"."int4_ops" ASC NULLS LAST
);
CREATE INDEX "idx_session_items_sort_order" ON "public"."session_items" USING btree (
  "sort_order" "pg_catalog"."int4_ops" ASC NULLS LAST
);
CREATE INDEX "idx_session_items_type" ON "public"."session_items" USING btree (
  "type" COLLATE "pg_catalog"."default" "pg_catalog"."text_ops" ASC NULLS LAST
);

-- ----------------------------
-- Primary Key structure for table session_items
-- ----------------------------
ALTER TABLE "public"."session_items" ADD CONSTRAINT "session_items_new_pkey" PRIMARY KEY ("id");

-- ----------------------------
-- Indexes structure for table sessions
-- ----------------------------
CREATE INDEX "idx_sessions_agent_config" ON "public"."sessions" USING gin (
  "agent_config" "pg_catalog"."jsonb_ops"
);
CREATE INDEX "idx_sessions_agent_id" ON "public"."sessions" USING btree (
  "agent_id" COLLATE "pg_catalog"."default" "pg_catalog"."text_ops" ASC NULLS LAST
);
CREATE INDEX "idx_sessions_context_config" ON "public"."sessions" USING gin (
  "context_config" "pg_catalog"."jsonb_ops"
);
CREATE INDEX "idx_sessions_tenant_id" ON "public"."sessions" USING btree (
  "tenant_id" "pg_catalog"."int4_ops" ASC NULLS LAST
);

-- ----------------------------
-- Primary Key structure for table sessions
-- ----------------------------
ALTER TABLE "public"."sessions" ADD CONSTRAINT "sessions_pkey" PRIMARY KEY ("id");

-- ----------------------------
-- Checks structure for table spatial_ref_sys
-- ----------------------------
ALTER TABLE "public"."spatial_ref_sys" ADD CONSTRAINT "spatial_ref_sys_srid_check" CHECK (srid > 0 AND srid <= 998999);

-- ----------------------------
-- Primary Key structure for table spatial_ref_sys
-- ----------------------------
ALTER TABLE "public"."spatial_ref_sys" ADD CONSTRAINT "spatial_ref_sys_pkey" PRIMARY KEY ("srid");

-- ----------------------------
-- Indexes structure for table tenants
-- ----------------------------
CREATE INDEX "idx_tenants_agent_config" ON "public"."tenants" USING gin (
  "agent_config" "pg_catalog"."jsonb_ops"
);
CREATE INDEX "idx_tenants_api_key" ON "public"."tenants" USING btree (
  "api_key" COLLATE "pg_catalog"."default" "pg_catalog"."text_ops" ASC NULLS LAST
);
CREATE INDEX "idx_tenants_status" ON "public"."tenants" USING btree (
  "status" COLLATE "pg_catalog"."default" "pg_catalog"."text_ops" ASC NULLS LAST
);

-- ----------------------------
-- Primary Key structure for table tenants
-- ----------------------------
ALTER TABLE "public"."tenants" ADD CONSTRAINT "tenants_pkey" PRIMARY KEY ("id");

-- ----------------------------
-- Indexes structure for table users
-- ----------------------------
CREATE INDEX "idx_users_deleted_at" ON "public"."users" USING btree (
  "deleted_at" "pg_catalog"."timestamptz_ops" ASC NULLS LAST
);
CREATE INDEX "idx_users_email" ON "public"."users" USING btree (
  "email" COLLATE "pg_catalog"."default" "pg_catalog"."text_ops" ASC NULLS LAST
);
CREATE INDEX "idx_users_tenant_id" ON "public"."users" USING btree (
  "tenant_id" "pg_catalog"."int4_ops" ASC NULLS LAST
);
CREATE INDEX "idx_users_username" ON "public"."users" USING btree (
  "username" COLLATE "pg_catalog"."default" "pg_catalog"."text_ops" ASC NULLS LAST
);

-- ----------------------------
-- Uniques structure for table users
-- ----------------------------
ALTER TABLE "public"."users" ADD CONSTRAINT "users_username_key" UNIQUE ("username");
ALTER TABLE "public"."users" ADD CONSTRAINT "users_email_key" UNIQUE ("email");

-- ----------------------------
-- Primary Key structure for table users
-- ----------------------------
ALTER TABLE "public"."users" ADD CONSTRAINT "users_pkey" PRIMARY KEY ("id");

-- ----------------------------
-- Foreign Keys structure for table auth_tokens
-- ----------------------------
ALTER TABLE "public"."auth_tokens" ADD CONSTRAINT "fk_auth_tokens_user" FOREIGN KEY ("user_id") REFERENCES "public"."users" ("id") ON DELETE CASCADE ON UPDATE NO ACTION;

-- ----------------------------
-- Foreign Keys structure for table users
-- ----------------------------
ALTER TABLE "public"."users" ADD CONSTRAINT "fk_users_tenant" FOREIGN KEY ("tenant_id") REFERENCES "public"."tenants" ("id") ON DELETE SET NULL ON UPDATE NO ACTION;

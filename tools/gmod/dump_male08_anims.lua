--[[
	dump_male08_anims.lua  (second pass)

	Records the real player animation poses of the job playermodel (one pose per
	rendered frame, so the engine actually re-evaluates the bones) and reads its
	textures back through a render target.

	Writes into garrysmod/data/:
		male08_anims.txt   bone matrices: idle per holdtype at many aim pitches, walk, run, crouch
		male08_tex.txt     the model's textures as base64 RGB

	Run it in SINGLE-PLAYER:
		sv_allowcslua 1
		lua_openscript_cl dump_male08_anims.lua
	then CLOSE the console (press ~) so the game keeps rendering, wait about 10 seconds
	and open it again: the last line should say "all done".
]]

local MODEL = "models/player/Suits/male_08_shirt_tie.mdl"
local OUT = "male08_anims.txt"
local TEX_OUT = "male08_tex.txt"
local HOOK = "male08_dump_anims"

local function P(fmt, ...)
	MsgC(Color(120, 220, 120), "[male08 dump] ", Color(230, 230, 230), string.format(fmt, ...), "\n")
end

local function num(n, decimals)
	if n ~= n then return "nan" end
	local s = string.format("%." .. decimals .. "f", n)
	s = s:gsub("0+$", ""):gsub("%.$", "")
	if s == "-0" then s = "0" end
	return s
end

local function mat12(m)
	local t = m:ToTable()
	local r = {}
	for row = 1, 3 do
		for col = 1, 4 do
			r[#r + 1] = num(t[row][col], col == 4 and 4 or 6)
		end
	end
	return table.concat(r, " ")
end

hook.Remove("PostDrawOpaqueRenderables", HOOK)
hook.Remove("PostRender", HOOK)

local ent = ClientsideModel(MODEL, RENDERGROUP_OPAQUE)
if not IsValid(ent) or ent:GetBoneCount() <= 1 then
	P("could not load %s - is the Suits addon mounted in this game?", MODEL)
	if IsValid(ent) then ent:Remove() end
	return
end
ent:SetNoDraw(true)
ent:SetPos(Vector(0, 0, 0))
ent:SetAngles(Angle(0, 0, 0))
local nb = ent:GetBoneCount()

-- pose jobs ---------------------------------------------------------------------

local jobs = {}
local function job(label, seq, cycle, params)
	jobs[#jobs + 1] = {label = label, seq = seq, cycle = cycle or 0, params = params or {}}
end

local pitches = {-80, -60, -45, -30, -15, 0, 15, 30, 45, 60, 80}
local main = {
	{"normal", "idle_all_01"}, {"passive", "idle_passive"}, {"fist", "idle_fist"}, {"pistol", "idle_pistol"},
	{"smg", "idle_smg1"}, {"ar2", "idle_ar2"}, {"melee", "idle_melee"},
}
for _, h in ipairs(main) do
	for _, pitch in ipairs(pitches) do
		job(string.format("aim_%s_%d", h[1], pitch), h[2], 0, {aim_pitch = pitch})
	end
end
for _, h in ipairs({{"shotgun", "idle_shotgun"}, {"melee2", "idle_melee2"}, {"knife", "idle_knife"},
		{"slam", "idle_slam"}, {"grenade", "idle_grenade"}, {"physgun", "idle_physgun"}, {"crossbow", "idle_crossbow"},
		{"rpg", "idle_rpg"}, {"revolver", "idle_revolver"}, {"camera", "idle_camera"}, {"magic", "idle_magic"},
		{"duel", "idle_dual"}}) do
	job("aim_" .. h[1] .. "_0", h[2], 0, {aim_pitch = 0})
end
for _, c in ipairs({0, 0.25, 0.5, 0.75}) do job("walk_" .. c, "walk_all", c, {move_x = 1}) end
for _, c in ipairs({0, 0.5}) do
	job("run_" .. c, "run_all_01", c, {move_x = 1})
	job("cwalk_" .. c, "cwalk_all", c, {move_x = 1})
end
job("crouch_idle", "cidle_all", 0, {})
job("crouch_pistol", "cidle_pistol", 0, {aim_pitch = 0})

local lines = {"male08anims 1", "model " .. ent:GetModel(), "bones " .. nb}
local function L(s) lines[#lines + 1] = s end

local function apply(j)
	local seq = ent:LookupSequence(j.seq)
	if not seq or seq < 0 then return false end
	if ent:GetSequence() ~= seq then ent:ResetSequence(seq) end
	ent:SetPlaybackRate(0)
	ent:SetCycle(j.cycle)
	for k = 0, ent:GetNumPoseParameters() - 1 do
		local name = ent:GetPoseParameterName(k)
		ent:SetPoseParameter(name, j.params[name] or 0)
	end
	return true
end

-- textures (read back through a render target) ------------------------------------

local tex_jobs = {}
do
	local meshes = util.GetModelMeshes(MODEL, 0, 0) or {}
	local seen = {}
	for _, m in ipairs(meshes) do
		local path = m.material
		if path and not seen[path] and not path:find("error", 1, true) then
			seen[path] = true
			local mat = Material(path)
			local base = mat:GetString("$basetexture")
			local tex = mat:GetTexture("$basetexture")
			if base and tex and not tex:IsError() then
				local w, h = tex:Width(), tex:Height()
				local limit = path:find("facemap", 1, true) and 512 or 256
				local s = math.min(1, limit / math.max(w, h))
				tex_jobs[#tex_jobs + 1] = {path = path, base = base, w = math.max(4, math.floor(w * s)), h = math.max(4, math.floor(h * s))}
			end
		end
	end
end

local tex_lines = {"male08tex 1"}

local function read_texture(t)
	local rt = GetRenderTarget("male08dump_rt_" .. t.w .. "x" .. t.h, t.w, t.h)
	local mat = CreateMaterial("male08dump_tex_" .. util.CRC(t.path), "UnlitGeneric", {["$basetexture"] = t.base})
	render.PushRenderTarget(rt, 0, 0, t.w, t.h)
	render.Clear(0, 0, 0, 255, true, true)
	cam.Start2D()
		surface.SetDrawColor(255, 255, 255, 255)
		surface.SetMaterial(mat)
		surface.DrawTexturedRect(0, 0, t.w, t.h)
	cam.End2D()
	render.CapturePixels()
	local rows = {}
	for y = 0, t.h - 1 do
		local row = {}
		for x = 0, t.w - 1 do
			local r, g, b = render.ReadPixel(x, y)
			row[#row + 1] = string.char(r or 0, g or 0, b or 0)
		end
		rows[#rows + 1] = table.concat(row)
	end
	render.PopRenderTarget()
	local data = util.Base64Encode(table.concat(rows), true):gsub("%s", "")
	tex_lines[#tex_lines + 1] = string.format("tex %s %d %d", t.path, t.w, t.h)
	tex_lines[#tex_lines + 1] = data
end

local function finish_textures()
	hook.Remove("PostRender", HOOK)
	local data = table.concat(tex_lines, "\n") .. "\n"
	file.Write(TEX_OUT, data)
	P("wrote garrysmod/data/%s (%d KB, %d textures)", TEX_OUT, math.ceil(#data / 1024), (#tex_lines - 1) / 2)
	P("all done - attach male08_anims.txt and male08_tex.txt from garrysmod/data")
end

local function start_textures()
	local k = 0
	hook.Add("PostRender", HOOK, function()
		k = k + 1
		local t = tex_jobs[k]
		if not t then
			finish_textures()
			return
		end
		local ok, err = pcall(read_texture, t)
		if ok then
			P("texture %d/%d: %s", k, #tex_jobs, t.path)
		else
			P("texture %s failed: %s", t.path, tostring(err))
			pcall(render.PopRenderTarget)
		end
	end)
end

-- one pose per frame: set the animation state, let the engine set up the bones, read them

local i, distinct, last_head = 0, 0, nil
local head = ent:LookupBone("ValveBiped.Bip01_Head1")

local function finish_poses()
	hook.Remove("PostDrawOpaqueRenderables", HOOK)
	ent:Remove()
	L("distinct " .. distinct)
	local data = table.concat(lines, "\n") .. "\n"
	file.Write(OUT, data)
	P("wrote garrysmod/data/%s (%d KB, %d poses, %d distinct)", OUT, math.ceil(#data / 1024), #jobs, distinct)
	if distinct < #jobs / 2 then
		P("WARNING: most poses came out identical - tell Claude")
	end
	start_textures()
end

hook.Add("PostDrawOpaqueRenderables", HOOK, function(depth, sky)
	if depth or sky then return end
	i = i + 1
	local j = jobs[i]
	if not j then
		finish_poses()
		return
	end
	if not apply(j) then
		L("nopose " .. j.label .. " " .. j.seq)
		return
	end
	ent:InvalidateBoneCache()
	ent:SetupBones()
	ent:DrawModel()

	local pp = {}
	for k, v in pairs(j.params) do pp[#pp + 1] = k .. "=" .. num(v, 2) end
	table.sort(pp)
	L(string.format("pose %s %s %s %s", j.label, j.seq, num(j.cycle, 3), #pp > 0 and table.concat(pp, ",") or "-"))
	for b = 0, nb - 1 do
		local m = ent:GetBoneMatrix(b)
		if m then L("pm " .. b .. " " .. mat12(m)) end
	end
	local hm = head and ent:GetBoneMatrix(head)
	if hm then
		local hp = hm:GetTranslation()
		if not last_head or hp:Distance(last_head) > 0.001 then distinct = distinct + 1 end
		last_head = hp
	end
	if i % 20 == 0 then P("pose %d/%d", i, #jobs) end
end)

P("recording %d poses and %d textures - close the console now and wait a few seconds", #jobs, #tex_jobs)

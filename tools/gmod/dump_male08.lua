--[[
	dump_male08.lua

	Exports the job playermodel so the PAC outfit can be fitted to it offline:
	skeleton, mesh (LOD 0, default bodygroups), attachments, flex names and the
	bone matrices of the real player animations (idle per holdtype at several aim
	pitches, walk, run, crouch). Also saves the model's base textures as jpgs.

	It only reads the model and writes files into garrysmod/data/:
		male08_dump.txt
		male08_tex_<n>.jpg

	Run it in SINGLE-PLAYER, in the console:
		sv_allowcslua 1
		lua_run_cl http.Fetch("<raw url of this file>", function(b) RunString(b, "dump_male08") end, print)
]]

local MODEL = "models/player/Suits/male_08_shirt_tie.mdl"
local OUT = "male08_dump.txt"

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

-- rows 1-3 of a VMatrix: columns are the bone's forward/left/up axes and its position
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

local lines = {}
local function L(s) lines[#lines + 1] = s end

local ent = ClientsideModel(MODEL, RENDERGROUP_OPAQUE)
if not IsValid(ent) then
	P("could not load %s - is the Suits addon mounted in this game?", MODEL)
	return
end
ent:SetPos(Vector(0, 0, 0))
ent:SetAngles(Angle(0, 0, 0))
ent:SetupBones()

local nb = ent:GetBoneCount()
if nb <= 1 then
	P("%s loaded as an error model (%d bones) - is the Suits addon mounted?", MODEL, nb)
	ent:Remove()
	return
end

L("male08dump 1")
L("model " .. ent:GetModel())
L("screen " .. ScrW() .. " " .. ScrH())
L("bones " .. nb)
for i = 0, nb - 1 do
	L(string.format("b %d %d %s", i, ent:GetBoneParent(i), ent:GetBoneName(i)))
end

for i = 0, ent:GetNumPoseParameters() - 1 do
	local min, max = ent:GetPoseParameterRange(i)
	L(string.format("pp %s %s %s", ent:GetPoseParameterName(i), num(min or 0, 3), num(max or 0, 3)))
end

if ent.GetFlexNum then
	for i = 0, ent:GetFlexNum() - 1 do
		local min, max = ent:GetFlexBounds(i)
		L(string.format("flex %d %s %s %s", i, ent:GetFlexName(i), num(min or 0, 3), num(max or 0, 3)))
	end
end

do
	local names = {}
	for _, name in pairs(ent:GetSequenceList() or {}) do names[#names + 1] = name end
	table.sort(names)
	L("sequences " .. #names)
	for i = 1, #names, 40 do
		L("seqnames " .. table.concat(names, " ", i, math.min(i + 39, #names)))
	end
end

-- mesh + bind pose ------------------------------------------------------------

local ok, err = pcall(function()
	local meshes, bind = util.GetModelMeshes(MODEL, 0, 0)
	if not meshes then L("nomesh") return end

	if bind then
		for k, b in pairs(bind) do
			if b.matrix then
				L(string.format("bind %s %s %s", tostring(k), tostring(b.parent or -1), mat12(b.matrix)))
			end
		end
	end

	for mi, m in ipairs(meshes) do
		local tris = m.triangles or {}
		L(string.format("mesh %d %d %s", mi, math.floor(#tris / 3), tostring(m.material)))

		local index, count, tri = {}, 0, {}
		for ti = 1, #tris do
			local v = tris[ti]
			local p, n = v.pos, v.normal or Vector(0, 0, 0)
			local parts = {
				num(p.x, 3), num(p.y, 3), num(p.z, 3),
				num(n.x, 3), num(n.y, 3), num(n.z, 3),
				num(v.u or 0, 4), num(v.v or 0, 4),
			}
			for _, w in ipairs(v.weights or {}) do
				parts[#parts + 1] = tostring(w.bone)
				parts[#parts + 1] = num(w.weight, 3)
			end
			local key = table.concat(parts, " ")

			local id = index[key]
			if not id then
				id = count
				count = count + 1
				index[key] = id
				L("v " .. key)
			end

			tri[#tri + 1] = id
			if #tri == 3 then
				L(string.format("t %d %d %d", tri[1], tri[2], tri[3]))
				tri = {}
			end
		end
	end
end)
if not ok then L("mesherror " .. tostring(err)) P("mesh export failed: %s", tostring(err)) end

-- animated poses ----------------------------------------------------------------

local function set_pose(seqname, cycle, params)
	local seq = isnumber(seqname) and seqname or ent:LookupSequence(seqname)
	if not seq or seq < 0 then return false end

	ent:ResetSequence(seq)
	ent:SetPlaybackRate(0)
	ent:SetCycle(cycle)
	for i = 0, ent:GetNumPoseParameters() - 1 do
		local name = ent:GetPoseParameterName(i)
		ent:SetPoseParameter(name, params[name] or 0)
	end
	ent:InvalidateBoneCache()
	ent:SetupBones()
	return true
end

local function dump_attachments(label)
	for _, a in ipairs(ent:GetAttachments() or {}) do
		local ap = ent:GetAttachment(a.id)
		if ap then
			L(string.format("att %s %s %s %s %s %s %s %s %s", label, a.name, tostring(ap.Bone or -1),
				num(ap.Pos.x, 3), num(ap.Pos.y, 3), num(ap.Pos.z, 3),
				num(ap.Ang.p, 3), num(ap.Ang.y, 3), num(ap.Ang.r, 3)))
		end
	end
end

local pose_count = 0
local function dump_pose(label, seqname, cycle, params)
	params = params or {}
	if not set_pose(seqname, cycle, params) then
		L("nopose " .. label .. " " .. tostring(seqname))
		return
	end

	local pp = {}
	for k, v in pairs(params) do pp[#pp + 1] = k .. "=" .. num(v, 2) end
	table.sort(pp)

	L(string.format("pose %s %s %s %s", label, isnumber(seqname) and ent:GetSequenceName(seqname) or seqname,
		num(cycle, 3), #pp > 0 and table.concat(pp, ",") or "-"))
	for i = 0, nb - 1 do
		local m = ent:GetBoneMatrix(i)
		if m then L("pm " .. i .. " " .. mat12(m)) end
	end
	pose_count = pose_count + 1
end

local ok2, err2 = pcall(function()
	dump_pose("seq0", 0, 0)
	dump_attachments("seq0")

	dump_pose("idle_normal_p0", "idle_all_01", 0, {aim_pitch = 0})
	dump_attachments("idle_normal_p0")

	local pitches = {-75, -45, -20, 0, 20, 45, 75}
	local main = {
		normal = "idle_all_01", passive = "idle_passive", fist = "idle_fist", pistol = "idle_pistol",
		smg = "idle_smg1", ar2 = "idle_ar2", melee = "idle_melee",
	}
	for _, holdtype in ipairs({"normal", "passive", "fist", "pistol", "smg", "ar2", "melee"}) do
		for _, pitch in ipairs(pitches) do
			dump_pose(string.format("aim_%s_%d", holdtype, pitch), main[holdtype], 0, {aim_pitch = pitch})
		end
	end

	local others = {
		shotgun = "idle_shotgun", melee2 = "idle_melee2", knife = "idle_knife", slam = "idle_slam",
		grenade = "idle_grenade", physgun = "idle_physgun", crossbow = "idle_crossbow", rpg = "idle_rpg",
		revolver = "idle_revolver", camera = "idle_camera", magic = "idle_magic", duel = "idle_dual",
	}
	for _, holdtype in ipairs({"shotgun", "melee2", "knife", "slam", "grenade", "physgun", "crossbow", "rpg",
			"revolver", "camera", "magic", "duel"}) do
		dump_pose("aim_" .. holdtype .. "_0", others[holdtype], 0, {aim_pitch = 0})
	end

	for _, c in ipairs({0, 0.25, 0.5, 0.75}) do
		dump_pose("walk_" .. c, "walk_all", c, {move_x = 1})
	end
	for _, c in ipairs({0, 0.5}) do
		dump_pose("run_" .. c, "run_all_01", c, {move_x = 1})
		dump_pose("cwalk_" .. c, "cwalk_all", c, {move_x = 1})
	end
	dump_pose("crouch_idle", "cidle_all", 0, {})
end)
if not ok2 then L("poseerror " .. tostring(err2)) P("pose export failed: %s", tostring(err2)) end

-- did the aim layers react to aim_pitch? (they should move the head)
pcall(function()
	local head = ent:LookupBone("ValveBiped.Bip01_Head1")
	if head and set_pose("idle_pistol", 0, {aim_pitch = -75}) then
		local a = ent:GetBoneMatrix(head):GetTranslation()
		set_pose("idle_pistol", 0, {aim_pitch = 75})
		local b = ent:GetBoneMatrix(head):GetTranslation()
		L("aimcheck " .. num(a:Distance(b), 3))
		P("aim pitch moves the head by %.2f units (should be well above 0)", a:Distance(b))
	end
end)

-- materials ---------------------------------------------------------------------

local textures = {}
local ok3, err3 = pcall(function()
	for i, path in ipairs(ent:GetMaterials() or {}) do
		local mat = Material(path)
		local base = mat and mat:GetString("$basetexture") or ""
		local tex = mat and mat:GetTexture("$basetexture")
		local w, h = 0, 0
		if tex and not tex:IsError() then w, h = tex:Width(), tex:Height() end
		L(string.format("mat %d %s %s %d %d", i, path, base ~= "" and base or "-", w, h))
		if base ~= "" and w >= 64 then
			textures[#textures + 1] = {index = i, name = base, w = w, h = h}
		end
	end
end)
if not ok3 then L("materror " .. tostring(err3)) end

ent:Remove()

local data = table.concat(lines, "\n") .. "\n"
file.Write(OUT, data)
P("wrote garrysmod/data/%s (%d KB, %d bones, %d poses)", OUT, math.ceil(#data / 1024), nb, pose_count)

-- textures are drawn to the top-left of the screen for one frame each and captured
if #textures > 0 then
	local i = 0
	hook.Add("PostRender", "male08_dump_textures", function()
		i = i + 1
		local t = textures[i]
		if not t then
			hook.Remove("PostRender", "male08_dump_textures")
			P("done - attach male08_dump.txt and the male08_tex_*.jpg files from garrysmod/data")
			return
		end

		local okt, errt = pcall(function()
			local size = math.min(1024, ScrW(), ScrH())
			local w, h = t.w, t.h
			local scale = math.min(1, size / math.max(w, h))
			w, h = math.floor(w * scale), math.floor(h * scale)

			local mat = CreateMaterial("male08_dump_tex_" .. t.index, "UnlitGeneric", {["$basetexture"] = t.name})
			cam.Start2D()
				surface.SetDrawColor(0, 0, 0, 255)
				surface.DrawRect(0, 0, w, h)
				surface.SetDrawColor(255, 255, 255, 255)
				surface.SetMaterial(mat)
				surface.DrawTexturedRect(0, 0, w, h)
			cam.End2D()

			local jpg = render.Capture({format = "jpeg", quality = 95, x = 0, y = 0, w = w, h = h})
			if jpg then
				file.Write("male08_tex_" .. t.index .. ".jpg", jpg)
				P("saved male08_tex_%d.jpg (%s)", t.index, t.name)
			end
		end)
		if not okt then P("texture %s failed: %s", t.name, tostring(errt)) end
	end)
else
	P("done - attach male08_dump.txt from garrysmod/data")
end

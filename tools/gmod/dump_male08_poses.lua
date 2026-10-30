--[[
	dump_male08_poses.lua  (third pass: animation poses only)

	The engine computes a clientside model's bones once and then keeps returning that
	cache, whatever sequence/cycle/pose parameters are set afterwards. So every pose here
	gets a brand-new model: set the animation state first, set up the bones once, read
	them, remove the model. Runs in one go, no frames involved.

	Writes ONE file: garrysmod/data/male08_dump3.txt

	Run it in SINGLE-PLAYER by pasting this one line into the console:
		sv_allowcslua 1; lua_run_cl http.Fetch("<raw url of this file>", function(b) RunString(b, "dump3") end)
]]

local MODEL = "models/player/Suits/male_08_shirt_tie.mdl"
local OUT = "male08_dump3.txt"

local function Notify(text)
	MsgC(Color(120, 220, 120), "[male08 export] ", Color(230, 230, 230), text, "\n")
	if notification then notification.AddLegacy(text, NOTIFY_GENERIC, 20) end
	if chat then chat.AddText(Color(120, 220, 120), "[male08 export] ", Color(255, 255, 255), text) end
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

local jobs = {}
local function job(label, seq, cycle, params)
	jobs[#jobs + 1] = {label = label, seq = seq, cycle = cycle or 0, params = params or {}}
end

local pitches = {-80, -60, -45, -30, -15, 0, 15, 30, 45, 60, 80}
for _, h in ipairs({{"normal", "idle_all_01"}, {"passive", "idle_passive"}, {"fist", "idle_fist"},
		{"pistol", "idle_pistol"}, {"smg", "idle_smg1"}, {"ar2", "idle_ar2"}, {"melee", "idle_melee"}}) do
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
-- in case the aim layers are not part of the idle sequences: the layers on their own
for _, layer in ipairs({"aimlayer_pistol", "aimlayer_ar2", "aimlayer_smg", "aimlayer_melee"}) do
	for _, pitch in ipairs({-80, 0, 80}) do
		job(string.format("layer_%s_%d", layer, pitch), layer, 0, {aim_pitch = pitch})
	end
end
-- what head_pitch does on its own
for _, h in ipairs({{"normal", "idle_all_01"}, {"pistol", "idle_pistol"}}) do
	for _, hp in ipairs({-60, -30, 30, 60}) do
		job(string.format("head_%s_%d", h[1], hp), h[2], 0, {aim_pitch = 0, head_pitch = hp})
	end
end

local lines = {"male08poses 1", "model " .. MODEL}
local head_z, distinct, nb = {}, 0, 0

for _, j in ipairs(jobs) do
	local ent = ClientsideModel(MODEL, RENDERGROUP_OPAQUE)
	if not IsValid(ent) then
		Notify("could not load " .. MODEL)
		return
	end
	ent:SetNoDraw(true)
	ent:SetPos(Vector(0, 0, 0))
	ent:SetAngles(Angle(0, 0, 0))
	nb = ent:GetBoneCount()
	local seq = ent:LookupSequence(j.seq)
	if seq and seq >= 0 then
		ent:ResetSequence(seq)
		ent:SetPlaybackRate(0)
		ent:SetCycle(j.cycle)
		for k = 0, ent:GetNumPoseParameters() - 1 do
			local name = ent:GetPoseParameterName(k)
			ent:SetPoseParameter(name, j.params[name] or 0)
		end
		ent:SetupBones()
		local pp = {}
		for k, v in pairs(j.params) do pp[#pp + 1] = k .. "=" .. num(v, 2) end
		table.sort(pp)
		lines[#lines + 1] = string.format("pose %s %s %s %s", j.label, j.seq, num(j.cycle, 3), #pp > 0 and table.concat(pp, ",") or "-")
		for b = 0, nb - 1 do
			local m = ent:GetBoneMatrix(b)
			if m then lines[#lines + 1] = "pm " .. b .. " " .. mat12(m) end
		end
		local head = ent:LookupBone("ValveBiped.Bip01_Head1")
		local hm = head and ent:GetBoneMatrix(head)
		if hm then
			local t = hm:GetTranslation()
			local key = string.format("%.2f %.2f %.2f", t.x, t.y, t.z)
			if not head_z[key] then head_z[key] = true distinct = distinct + 1 end
		end
	else
		lines[#lines + 1] = "nopose " .. j.label .. " " .. j.seq
	end
	ent:Remove()
end

lines[#lines + 1] = "distinct " .. distinct
local data = table.concat(lines, "\n") .. "\n"
file.Write(OUT, data)
if distinct < 10 then
	Notify(string.format("wrote %s but only %d different poses came out - tell Claude", OUT, distinct))
else
	Notify(string.format("done! %d poses (%d different). Attach garrysmod/data/%s - same folder as male08_dump.txt", #jobs, distinct, OUT))
end

--[[
	dump_male08_sit.lua  (fourth pass: the sitting animations)

	Same method as dump_male08_poses.lua (a brand-new model per pose, animation state set
	before the bones are built once), for the poses a seated player plays: the chair sit
	(ACT_HL2MP_SIT, used by spawn-menu chairs and sit-anywhere addons), the vehicle-seat
	sit_rollercoaster, sit_zen, and the weapon variants (sit_pistol, ...), with the head
	looking level, up and down.

	Writes ONE file: garrysmod/data/male08_dump4.txt

	Run it in SINGLE-PLAYER by pasting this one line into the console:
		sv_allowcslua 1; lua_run_cl http.Fetch("<raw url of this file>", function(b) RunString(b, "dump4") end)
]]

local MODEL = "models/player/Suits/male_08_shirt_tie.mdl"
local OUT = "male08_dump4.txt"

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
local function job(label, seq, cycle, params, act)
	jobs[#jobs + 1] = {label = label, seq = seq, cycle = cycle or 0, params = params or {}, act = act}
end

-- the chair sit by activity (what the seat code asks for), and by name in case the model names it
for _, hp in ipairs({-30, 0, 30}) do
	job("sit_act_head" .. hp, "ACT_HL2MP_SIT", 0, {aim_pitch = 0, head_pitch = hp}, ACT_HL2MP_SIT)
end
for _, name in ipairs({"sit", "sit_rollercoaster", "sit_zen", "sit_pistol", "sit_smg1", "sit_ar2", "sit_shotgun",
		"sit_melee", "sit_melee2", "sit_fist", "sit_passive", "sit_knife", "sit_grenade", "sit_physgun",
		"sit_rpg", "sit_crossbow", "sit_camera", "sit_slam", "sit_duel", "sit_revolver", "sit_magic", "drive_pd"}) do
	job(name, name, 0, {aim_pitch = 0})
end
for _, c in ipairs({0.25, 0.5, 0.75}) do job("sit_c" .. c, "sit", c, {aim_pitch = 0}) end

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
	local seq = j.act and ent:SelectWeightedSequence(j.act) or ent:LookupSequence(j.seq)
	if j.act and seq and seq >= 0 then j.seq = ent:GetSequenceName(seq) end
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
if distinct < 4 then
	Notify(string.format("wrote %s but only %d different poses came out - tell Claude", OUT, distinct))
else
	Notify(string.format("done! %d poses (%d different). Attach garrysmod/data/%s ", #jobs, distinct, OUT))
end

-- SCONE script for a high-jump measure.
-- See Tutorial 6a - Script - High Jump

function init( model, par, side )
	-- get the 'target_body' parameter from ScriptMeasure, or set to "pelvis"	
	target_body = scone.target_body or "pelvis"
	target_body_pos = vec3:new(0, tonumber(scone.target_body_y) or 0.0, 0)
	body = model:find_body( target_body )

	show_target = (tonumber(scone.show_target) or 1.0) > 0
	target_duration = tonumber(scone.target_duration) or 2.0
	extend_last_target_duration = (tonumber(scone.extend_last_target_duration) or 0.0) > 0
	react_time = tonumber(scone.react_time) or 0.5
	vel_penalty = tonumber(scone.vel_penalty) or 0.0
	ang_vel_weight = tonumber(scone.ang_vel_weight) or 0.25
	target_start = -target_duration

	targets = {
		vec3:new(-0.07, 1.88, 0.18) -- up
		-- vec3:new(0.72, 1.26, 0.15), -- fr
		-- vec3:new(-0.17, 0.91, -0.1), -- bh
		-- vec3:new(0.0, 1.34, 0.85), -- sd
	}
	cur_trg_idx = 0

	pos_tot = 0
	vel_tot = 0
	active_time = 0
	active = false

	body_pos = body:point_pos(target_body_pos)
end

function update( model, time )
	local t = time - target_start
	local dt = model:delta_time()
	if cur_trg_idx < #targets and t >= target_duration then
		cur_trg_idx = cur_trg_idx + 1
		target_pos = targets[cur_trg_idx]
		target_start = time
		t = 0.0
		if show_target then
			target_object = model:find_body("target")
			target_object:set_com_pos(target_pos)
		end
	end


	-- compute dist
	body_pos = body:point_pos(target_body_pos)
	body_vel = body:point_vel(target_body_pos)
	cur_delta = target_pos - body_pos
	lin_dist = cur_delta:length()
	lin_vel = body_vel:length()
	ang_vel = body:ang_vel():length()
	
	-- update penalty
	local keep_going = extend_last_target_duration and cur_trg_idx == #targets
	active = t >= react_time and ( keep_going or t <= target_duration )
	if active then
		pos_tot = pos_tot + dt * lin_dist 
		vel_tot = vel_tot + dt * ( lin_vel + ang_vel_weight * ang_vel )
		active_time = active_time + dt
		cur_pen = lin_dist + vel_penalty * lin_vel
	else
		cur_pen = 0.0
	end

	return false
end

function current_fitness()
	if active_time > 0 then
		return ( pos_tot + vel_penalty * vel_tot ) / active_time
	else
		return 0.0
	end
end


function result( model )
	-- this is called at the end of the simulation
	-- fitness corresponds to average body height
	return current_fitness()
end

function store_data( frame )
	-- store some values for analysis
	if lin_dist then
		frame:set_value( "reach.distance", lin_dist )
		frame:set_value( "reach.lin_vel", lin_vel )
		frame:set_value( "reach.ang_vel", ang_vel )
		frame:set_value( "reach.penalty", cur_pen )
		frame:set_value( "reach.active_time", active_time )
		frame:set_value( "reach.fitness", current_fitness() )
	end
end

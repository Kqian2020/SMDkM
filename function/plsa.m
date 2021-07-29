function [prob_term_topic, prob_topic_doc, lls] = plsa(termDocMatrix, numTopic, iter)
% PLSA model
[numTerm, numDoc] = size(termDocMatrix);

prob_term_topic = rand(numTerm, numTopic); % p(term | topic)
for i = 1:numTopic
	prob_term_topic(:, i) = prob_term_topic(:, i) / sum(prob_term_topic(:, i));
end

prob_topic_doc = rand(numTopic, numDoc);   % p(topic | doc)
for i = 1:numDoc
	prob_topic_doc(:, i) = prob_topic_doc(:, i)/ sum(prob_topic_doc(:, i) + 0.2);
end

prob_term_doc = zeros(numTerm, numDoc);
for d = 1:numDoc
	for z = 1:numTopic
		prob_term_doc(:, d) = prob_term_doc(:, d) + ...
		prob_topic_doc(z, d) .* prob_term_topic(:, z);
	end
	assert(sum(prob_term_doc(:, d)) - 1.0 < 1e-6);
end

prob_topic_term_doc = cell(numTopic, 1);   % p(topic | doc, term)
for z = 1 : numTopic
	prob_topic_term_doc{z} = zeros(numTerm, numDoc);
end


lls = []; % maximum log-likelihood estimations

for i = 1 : iter
% 	E-step;
	for d = 1:numDoc
		w = find(termDocMatrix(:, d));
		for z = 1:numTopic
			prob_topic_term_doc{z}(w, d) = prob_topic_doc(z, d) .* prob_term_topic(w, z) ./ prob_term_doc(w, d);
		end
	end
	
% 	M-step;
	for d = 1:numDoc
		w = find(termDocMatrix(:, d));
		for z = 1:numTopic
			prob_topic_doc(z, d) = sum(termDocMatrix(w, d) .* prob_topic_term_doc{z}(w, d));
		end
		prob_topic_doc(:, d) = prob_topic_doc(:, d) / sum(prob_topic_doc(:, d) + 0.2);
    end
	for z = 1:numTopic
		for w = 1:numTerm
			d = find(termDocMatrix(w, :));
			prob_word_topic(w, z) = sum(termDocMatrix(w, d) .* prob_topic_term_doc{z}(w, d)) ;
		end
		prob_word_topic(:, z) = prob_word_topic(:,z) / sum(prob_word_topic(:,z));
	end
	
	% calculate likelihood and update p(term, doc)
	for d = 1:numDoc
		prob_term_doc(:, d) = 0;
		for z = 1:numTopic
			prob_term_doc(:, d) = prob_term_doc(:, d) + ...
			prob_topic_doc(z, d) .* prob_term_topic(:, z);
		end
		assert((sum(prob_term_doc(:, d)) - 1.0) < 1e6);
    end
end
end

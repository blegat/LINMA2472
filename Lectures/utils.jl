begin
    # /!\ Important: use this Zotero config with Better BibTeX:
    # https://github.com/JuliaDocs/DocumenterCitations.jl/issues/85#issuecomment-2479025454
    function load_biblio!(file = joinpath(@__DIR__, "biblio.bib"); style = DocumenterCitations.AlphaStyle())
        @info("Loading bibliography from `$file`...")
        biblio = DocumenterCitations.CitationBibliography(file; style)
        DocumenterCitations.init_bibliography!(style, biblio)
        @info("Loading completed.")
        return biblio
    end
    citation_label(biblio, key::String) = DocumenterCitations.citation_label(biblio.style, biblio.entries[key], biblio.citations)
    function bibcite(biblio, keys::Vector{String})
        return "[" * join(citation_label.(Ref(biblio), keys), ", ") * "]"
    end
    bibcite(biblio, key::String) = bibcite(biblio, [key])
    function bibcite(biblio, key::String, what)
        return "[" * citation_label(biblio, key) * "; " * what * "]"
    end
    function citation_reference(biblio, key::String)
        # `DocumenterCitations` writes a `+` in the label after 3 authors so we use
        # `et_al = 3` for consistency
        DocumenterCitations.format_labeled_bibliography_reference(biblio.style, biblio.entries[key], et_al = 3)
    end
    # Markdown creates a `<p>` surrounding it but we don't want that in some cases
    _inline_markdown(m::Markdown.MD) = sprint(Markdown.htmlinline, m.content[].content)
    function _print_entry(io, biblio, key; links = false, kws...)
        print(io, '[')
        print(io, citation_label(biblio, key))
        print(io, "] ")
        println(io, _inline_markdown(Markdown.parse(citation_reference(biblio, key))))
    end
	function bibrefs(biblio, key::String; kws...)
		io = IOBuffer()
		println(io, "<p style=\"font-size:12px\">")
		_print_entry(io, biblio, key; kws...)
		println(io, "</p>")
		return HTML(String(take!(io)))
	end
	function bibrefs(biblio, keys::Vector{String}; kws...)
		io = IOBuffer()
		println(io, "<p style=\"font-size:12px\">")
		for key in keys
			_print_entry(io, biblio, key; kws...)
			println(io, "<br/>")
		end
		println(io, "</p>")
		return HTML(String(take!(io)))
	end

    function CenteredBoundedBox(str)
        xbearing, ybearing, width, height, xadvance, yadvance =
            Luxor.textextents(str)
        lcorner = Point(xbearing - width/2, ybearing)
        ocorner = Point(lcorner.x + width, lcorner.y + height)
        return BoundingBox(lcorner, ocorner)
    end
    function boxed(str::AbstractString, p)
        translate(p)
        sethue("lightgrey")
        poly(CenteredBoundedBox(str) + 5, action = :stroke, close=true)
        sethue("black")
        text(str, Point(0, 0), halign=:center)
        #settext("<span font='26'>$str</span>", halign="center", markup=true)
        origin()
    end

    # `Cols` conflict with `DataFrames`
    struct HAlign{T<:Tuple}
        cols::T
        dims::Vector{Int}
    end
    function HAlign(a::Tuple)
        n = length(a)
        return HAlign(a, div(100, n) * ones(Int, n))
    end
    HAlign(a, b, args...) = HAlign(tuple(a, b, args...))

    function Base.show(io, mime::MIME"text/html", c::HAlign)
        x = div(100, length(c.cols))
    	write(io, """<div style="display: flex; justify-content: center; align-items: center;">""")
        for (col, p) in zip(c.cols, c.dims)
            write(io, """<div style="flex: $p%;">""")
            show(io, mime, col)
            write(io, """</div>""")
        end
    	write(io, """</div>""")
    end
    function imgpath(file)
        if !('.' in file)
            file = file * ".png"
        end
        return joinpath(joinpath(@__DIR__, "images", file))
    end
    function img(file, args...)
        LocalResource(imgpath(file), args...)
    end
    section(t) = md"# $t"
    frametitle(t) = md"# $t" # with `##`, it's not centered

    struct Join
		list
	    Join(a) = new(a)
	    Join(a, b, args...) = Join(tuple(a, b, args...))
    end
	function Base.show(io::IO, mime::MIME"text/html", d::Join)
		for el in d.list
			show(io, mime, el)
		end
	end

	struct HTMLTag
		tag::String
		parent
	end
	function Base.show(io::IO, mime::MIME"text/html", d::HTMLTag)
		write(io, "<", d.tag, ">")
		show(io, mime, d.parent)
		write(io, "</", d.tag, ">")
	end

    function qa(question, answer)
        return HTMLTag("details", Join(HTMLTag("summary", question), answer))
    end

    function qa(question::Markdown.MD, answer)
        # `html(question)` will create `<p>` because `question.content[]` is `Markdown.Paragraph`
        # This will print the question on a new line and we don't want that:
        h = HTML(sprint(Markdown.htmlinline, question.content[].content))
        return qa(h, answer)
    end
end
